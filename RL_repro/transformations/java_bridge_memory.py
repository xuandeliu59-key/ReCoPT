
import subprocess
import time
import threading
import datetime
import os
import jpype


class JavaDaemon:
    def __init__(self, jar_path, log_prefix="Daemon"):
        self.jar_path = jar_path
        self.log_prefix = log_prefix
        self.error_log_file = "java_error_log.txt"
        self.call_count = 0  
        self.process = None
        
        self._start_process()

    def _start_process(self):
        self.process = subprocess.Popen(
            [
                "java", 
                "-Xms1G", "-Xmx2G", 
                "-XX:+UseG1GC",             
                "-XX:MaxGCPauseMillis=50",  
                "-Dfile.encoding=UTF-8", 
                "-Dorg.slf4j.simpleLogger.defaultLogLevel=off", 
                "-jar", self.jar_path
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, 
            text=True, encoding="utf-8", bufsize=1
        )
        time.sleep(1)

    def _restart_process(self):
        
        self.close()
        self._start_process()

    def execute_rule(self, code_string, rule_id, timeout=5.0):
        
        self.call_count += 1
        if self.call_count > 20000:
            self._restart_process()
            self.call_count = 0
            
        try:
            result_container = {"output": None}

            def interact():
                try:
                    self.process.stdin.write(f"{rule_id}\n")
                    self.process.stdin.write(code_string)
                    if not code_string.endswith('\n'):
                        self.process.stdin.write("\n")
                    self.process.stdin.write("===END_OF_CODE===\n")
                    self.process.stdin.flush()

                    result_lines = []
                    while True:
                        line = self.process.stdout.readline()
                        if not line or "===END_OF_RESULT===" in line:
                            break
                        if line.startswith("ERROR:"):
                            result_container["output"] = code_string
                            return
                        if "SLF4J" in line or "[main] DEBUG" in line:
                            continue
                        result_lines.append(line)
                        
                    result_container["output"] = "".join(result_lines).strip()
                except Exception:
                    pass

            thread = threading.Thread(target=interact)
            thread.start()
            thread.join(timeout=timeout) 

            if thread.is_alive():
                self._restart_process()     
                return code_string        

            return result_container["output"] if result_container["output"] is not None else code_string

        except Exception:
            self._restart_process()
            return code_string
        
    def close(self):
        if self.process:
            try:
                self.process.terminate() 
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait() 
            except Exception:
                pass
            finally:
                
                try:
                    if self.process.stdin: self.process.stdin.close()
                    if self.process.stdout: self.process.stdout.close()
                except Exception:
                    pass
            self.process = None

class OpenRewriteJPypeEngine:
    def __init__(self, rewrite_jar, spat_jar=None, log_prefix="JPypeEngine"):
        self.log_prefix = log_prefix
        self.error_log_file = "java_error_log.txt"
        self.call_count = 0  

        if not jpype.isJVMStarted():
            classpath = [rewrite_jar]
            if spat_jar:
                classpath.append(spat_jar)

            
            
            
            _saved_ld = os.environ.pop('LD_LIBRARY_PATH', None)

            
            jpype.startJVM(
                jpype.getDefaultJVMPath(),
                "-Xms1G", "-Xmx2G", 
                "-XX:MaxMetaspaceSize=256M",
                "-XX:+UseG1GC",
                "-XX:MinHeapFreeRatio=10",
                "-XX:MaxHeapFreeRatio=20",
                "-XX:-ShrinkHeapInSteps",
                "-Dfile.encoding=UTF-8",
                classpath=classpath
            )

            
            if _saved_ld is not None:
                os.environ['LD_LIBRARY_PATH'] = _saved_ld

        self.JavaParser = jpype.JClass("org.openrewrite.java.JavaParser")
        self.InMemoryExecutionContext = jpype.JClass("org.openrewrite.InMemoryExecutionContext")
        self.InMemoryLargeSourceSet = jpype.JClass("org.openrewrite.internal.InMemoryLargeSourceSet")
        self.JClass = jpype.JClass("java.lang.Class")
        self.JString = jpype.JClass("java.lang.String")
        self.System = jpype.JClass("java.lang.System") 

        self.ctx = self.InMemoryExecutionContext()
        self.parser = self.JavaParser.fromJavaVersion().build()
        self.recipe_cache = {}

    def _soft_restart(self):
        try:
            print(f"[{self.log_prefix}] Restarting (call_count={self.call_count})")

            
            if hasattr(self, 'parser'):
                del self.parser
            self.parser = self.JavaParser.fromJavaVersion().build()

            self.recipe_cache.clear()

            if hasattr(self, 'ctx'):
                del self.ctx
            self.ctx = self.InMemoryExecutionContext()

            self.System.gc()
            self.System.runFinalization()
            self.System.gc()
            print(f"[{self.log_prefix}] Restart complete")
        except Exception:
            pass

    def execute_rule(self, code_string, recipe_name):
        self.call_count += 1
        if self.call_count > 20000:
            self._soft_restart()
            self.call_count = 0

        try:
            if recipe_name not in self.recipe_cache:
                recipe_class = self.JClass.forName(recipe_name)
                recipe_instance = recipe_class.getDeclaredConstructor().newInstance()
                self.recipe_cache[recipe_name] = recipe_instance
            recipe = self.recipe_cache[recipe_name]

            java_string = self.JString(code_string)
            source_files = self.parser.parse(self.ctx, java_string).toList()
            source_set = self.InMemoryLargeSourceSet(source_files)

            
            recipe_run = recipe.run(source_set, self.ctx)
            changeset = recipe_run.getChangeset()
            results = changeset.getAllResults()

            final_code = code_string
            if not results.isEmpty():
                result0 = results.get(0)
                after = result0.getAfter()
                final_code = str(after.printAll())
                del after
                del result0

            del results
            del changeset
            del recipe_run
            del source_set
            del source_files
            del java_string

            return final_code

        except Exception as e:
            with open(self.error_log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.datetime.now()}] JPype error: {e}\nRule: {recipe_name}\n")
            return code_string

    def close(self):
        pass


OPENREWRITE_RECIPES = {
    18: "org.openrewrite.staticanalysis.FinalizeLocalVariables",
    19: "org.openrewrite.staticanalysis.NeedBraces",
    20: "org.openrewrite.staticanalysis.UseDiamondOperator",
    21: "org.openrewrite.staticanalysis.RemoveExtraSemicolons",
    22: "org.openrewrite.staticanalysis.EmptyBlock",
    23: "org.openrewrite.staticanalysis.SimplifyBooleanReturn",
    24: "org.openrewrite.staticanalysis.UseJavaStyleArrayDeclarations",
    25: "org.openrewrite.staticanalysis.UnnecessaryParentheses",
    26: "org.openrewrite.staticanalysis.SimplifyBooleanExpression",
    27: "org.openrewrite.staticanalysis.ModifierOrder",
    28: "org.openrewrite.staticanalysis.ExplicitInitialization",
    29: "org.openrewrite.staticanalysis.ControlFlowIndentation",
    30: "org.openrewrite.staticanalysis.SimplifyCompoundStatement",
    31: "org.openrewrite.staticanalysis.SimplifyConstantIfBranchExecution",
    32: "org.openrewrite.staticanalysis.NoDoubleBraceInitialization",
    33: "org.openrewrite.staticanalysis.UnnecessaryThrows",
    34: "org.openrewrite.staticanalysis.InlineVariable",
    35: "org.openrewrite.staticanalysis.InstanceOfPatternMatch",
    36: "org.openrewrite.staticanalysis.UseLambdaForFunctionalInterface",
    37: "org.openrewrite.staticanalysis.UseForEachRemoveInsteadOfSetRemoveAll",

    38: "org.openrewrite.staticanalysis.UnwrapRepeatableAnnotations",
    39: "org.openrewrite.staticanalysis.CatchClauseOnlyRethrows",
    40: "org.openrewrite.staticanalysis.RemoveRedundantTypeCast",
    41: "org.openrewrite.staticanalysis.RemoveUnneededBlock",
    42: "org.openrewrite.staticanalysis.RemoveUnusedPrivateFields",
    43: "org.openrewrite.staticanalysis.RemoveUnusedPrivateMethods",
    44: "org.openrewrite.staticanalysis.UnnecessaryCloseInTryWithResources",
    45: "org.openrewrite.staticanalysis.UnnecessaryExplicitTypeArguments",
    46: "org.openrewrite.staticanalysis.SimplifyConsecutiveAssignments",
    47: "org.openrewrite.staticanalysis.DefaultComesLast",
    48: "org.openrewrite.staticanalysis.FallThrough",
    49: "org.openrewrite.staticanalysis.FinalizeMethodArguments",
    50: "org.openrewrite.staticanalysis.FinalizePrivateFields",
    51: "org.openrewrite.staticanalysis.FixStringFormatExpressions",
    52: "org.openrewrite.staticanalysis.HiddenField",
    53: "org.openrewrite.staticanalysis.HideUtilityClassConstructor",
    54: "org.openrewrite.staticanalysis.NestedEnumsAreNotStatic",
    55: "org.openrewrite.staticanalysis.OperatorWrap",
    56: "org.openrewrite.staticanalysis.RedundantFileCreation",
    57: "org.openrewrite.staticanalysis.ReplaceOptionalIsPresentWithIfPresent",
    58: "org.openrewrite.staticanalysis.ReplaceDeprecatedRuntimeExecMethods",
    59: "org.openrewrite.staticanalysis.SortedSetStreamToLinkedHashSet",
    60: "org.openrewrite.staticanalysis.UpperCaseLiteralSuffixes",
    61: "org.openrewrite.staticanalysis.UseSystemLineSeparator",
    62: "org.openrewrite.staticanalysis.AvoidBoxedBooleanExpressions",
    63: "org.openrewrite.staticanalysis.ChainStringBuilderAppendCalls",
    64: "org.openrewrite.staticanalysis.CovariantEquals",
    65: "org.openrewrite.staticanalysis.EqualsAvoidsNull",
    66: "org.openrewrite.staticanalysis.NoPrimitiveWrappersForToStringOrCompareTo",
    67: "org.openrewrite.staticanalysis.IndexOfShouldNotCompareGreaterThanZero",
    68: "org.openrewrite.staticanalysis.IndexOfReplaceableByContains"
}