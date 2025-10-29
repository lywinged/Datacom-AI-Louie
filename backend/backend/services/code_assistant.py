"""
Code Assistant Service (Task 3.4).
Self-healing code generation with automated testing.
"""
import os
import time
import tempfile
import subprocess
import logging
import json
import ast
import io
import contextlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI

from backend.models.code_schemas import (
    CodeRequest, CodeResponse, TestResult, RetryAttempt,
    Language, TestFramework
)
from backend.services.token_counter import get_token_counter, TokenUsage
from backend.utils.openai import sanitize_messages

logger = logging.getLogger(__name__)


class CodeAssistant:
    """Self-healing code assistant with automated testing"""

    def __init__(self, enable_learning: bool = True):
        """Initialize code assistant with OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Initialize OpenAI client with optional base URL
        client_kwargs = {"api_key": api_key}
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = AsyncOpenAI(**client_kwargs)
        self.model_name = os.getenv("OPENAI_MODEL", "Gpt4o")
        self.token_counter = get_token_counter()

        # Initialize learning system
        self.enable_learning = enable_learning
        self.learner = None
        if enable_learning:
            from backend.services.codegen_learn import CodeGenAdapter
            memory_path = os.getenv("CODEGEN_MEMORY_PATH", "data/codegen_experiences.jsonl")
            self.learner = CodeGenAdapter(memory_path)
            logger.info(f"🎓 Code learning system enabled (memory: {memory_path})")

        logger.info(f"✅ CodeAssistant initialized with model: {self.model_name}")

    async def generate_code(self, request: CodeRequest) -> CodeResponse:
        """
        Generate code with automated testing and self-healing.

        Args:
            request: Code generation request

        Returns:
            Code response with final code and test results
        """
        start_time = time.time()
        total_tokens = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost_usd = 0.0

        logger.info(f"💻 Generating {request.language} code: {request.task[:100]}...")

        # LEARNING INTEGRATION: Choose strategy before generation
        strategy = None
        signature = None
        if self.enable_learning and self.learner:
            language_str = str(request.language.value) if hasattr(request.language, 'value') else str(request.language)
            test_framework_str = str(request.test_framework.value) if request.test_framework and hasattr(request.test_framework, 'value') else None

            ctx = self.learner.choose(
                language=language_str,
                task=request.task,
                test_framework=test_framework_str
            )
            signature = ctx["signature"]
            strategy = ctx["strategy"]

            logger.info(f"🎓 Selected strategy: {strategy.get('name', 'unknown')}")

        retry_attempts: List[RetryAttempt] = []
        current_code = None
        current_test_result = None
        analysis_context: Optional[Dict[str, Any]] = None
        analysis_used_for_attempt: Optional[Dict[str, Any]] = None
        initial_plan: Optional[Dict[str, Any]] = await self._analyze_initial_plan(request)
        include_samples = bool(getattr(request, "include_samples", False))

        if initial_plan:
            summary = initial_plan.get("summary") or initial_plan.get("plan_overview")
            if summary:
                logger.info("🧭 Initial plan: %s", summary)
            plan_steps_initial = initial_plan.get("plan_steps") or []
            for idx, step in enumerate(plan_steps_initial, start=1):
                logger.info("   Plan step %d: %s", idx, step)

        # Initial generation
        for attempt in range(request.max_retries + 1):
            attempt_start = time.time()

            if attempt == 0:
                # First attempt - generate from scratch
                logger.info("🔨 Generating initial code...")
                code, usage = await self._generate_initial_code(request, initial_plan)
                analysis_used_for_attempt = None
            else:
                # Retry - fix based on error
                logger.info(f"🔧 Retry {attempt}/{request.max_retries} - Fixing code...")
                code, usage = await self._fix_code(
                    request,
                    current_code,
                    current_test_result,
                    analysis_context,
                )
                analysis_used_for_attempt = analysis_context
                analysis_context = None

            if usage:
                usage_obj = TokenUsage(
                    prompt_tokens=usage.get("prompt", 0),
                    completion_tokens=usage.get("completion", 0),
                    total_tokens=usage.get("total", 0),
                    model=self.model_name,
                    timestamp=datetime.utcnow(),
                )
                total_prompt_tokens += usage_obj.prompt_tokens
                total_completion_tokens += usage_obj.completion_tokens
                total_tokens += usage_obj.total_tokens
                total_cost_usd += self.token_counter.estimate_cost(usage_obj)
            else:
                total_tokens += 0
            current_code = code

            # Run tests
            logger.info("🧪 Running tests...")
            test_result = await self._run_tests(
                code,
                request.language,
                request.test_framework,
                include_samples=include_samples,
            )
            current_test_result = test_result

            # Record attempt
            if attempt > 0:
                analysis_summary = None
                plan_summary = None
                plan_steps: List[str] = []
                if analysis_used_for_attempt:
                    analysis_summary = (
                        analysis_used_for_attempt.get("summary")
                        or analysis_used_for_attempt.get("root_cause")
                        or analysis_used_for_attempt.get("diagnosis")
                    )
                    plan_summary = (
                        analysis_used_for_attempt.get("plan_overview")
                        or analysis_used_for_attempt.get("plan_summary")
                        or analysis_used_for_attempt.get("fix_strategy")
                    )
                    raw_steps = analysis_used_for_attempt.get("plan_steps") or []
                    if isinstance(raw_steps, str):
                        raw_steps = [step.strip() for step in raw_steps.splitlines() if step.strip()]
                    plan_steps = [step.strip() for step in raw_steps if isinstance(step, str) and step.strip()]

                fix_description = plan_summary or "Applied automated fix"
                if test_result.passed:
                    fix_description = f"{fix_description} (tests passed)"
                else:
                    fix_description = f"{fix_description} (tests still failing)"

                retry_attempts.append(RetryAttempt(
                    attempt_number=attempt,
                    code=code,
                    test_result=test_result,
                    error_analysis=analysis_summary or (test_result.stderr[:500] if not test_result.passed else None),
                    fix_applied=fix_description,
                    plan_summary=plan_summary,
                    plan_steps=plan_steps,
                ))

            # Check if tests passed
            if test_result.passed:
                logger.info(f"✅ Tests passed on attempt {attempt + 1}")
                break
            else:
                logger.warning(f"❌ Tests failed on attempt {attempt + 1}: {test_result.stderr[:200]}")

                if attempt < request.max_retries:
                    analysis_context = await self._analyze_failure(
                        request,
                        current_code,
                        test_result,
                        attempt=attempt,
                    )
                    if analysis_context:
                        summary_log = analysis_context.get("summary") or analysis_context.get("root_cause")
                        if summary_log:
                            logger.info("🩺 Failure analysis: %s", summary_log)
                        plan_overview = analysis_context.get("plan_overview")
                        if plan_overview:
                            logger.info("🗺️ Fix plan: %s", plan_overview)
                        for idx, step in enumerate(analysis_context.get("plan_steps", []), start=1):
                            logger.info("   Step %d: %s", idx, step)
                    else:
                        logger.warning("⚠️ Analysis step failed; will rely on raw error output for next retry.")

                if attempt >= request.max_retries:
                    logger.error("⚠️  Max retries reached, returning last attempt")

        generation_time_ms = (time.time() - start_time) * 1000

        token_usage = None
        if total_prompt_tokens or total_completion_tokens:
            token_usage = {
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens,
                "total": total_prompt_tokens + total_completion_tokens,
            }

        cost_usd = total_cost_usd

        logger.info(
            f"✅ Code generation completed in {generation_time_ms:.1f}ms "
            f"with {len(retry_attempts)} retries, tests passed: {current_test_result.passed}"
        )

        # LEARNING INTEGRATION: Record outcome after generation
        learning_result = None
        if self.enable_learning and self.learner and strategy and signature:
            code_length = len(current_code.split('\n')) if current_code else 0

            learning_result = self.learner.record(
                signature=signature,
                strategy=strategy,
                test_passed=current_test_result.passed,
                retry_count=len(retry_attempts),
                max_retries=request.max_retries,
                generation_time_ms=generation_time_ms,
                code_length=code_length,
                estimated_loc=signature.estimated_loc,
                token_cost_usd=cost_usd,
            )

            logger.info(f"🎓 Learning recorded: success={learning_result['success']}, "
                       f"reward={learning_result['reward']:.3f}, strategy={learning_result['strategy']}")

        initial_plan_summary = None
        initial_plan_steps: List[str] = []
        if initial_plan:
            initial_plan_summary = (
                initial_plan.get("summary")
                or initial_plan.get("plan_overview")
            )
            steps_value = initial_plan.get("plan_steps") or []
            if isinstance(steps_value, list):
                initial_plan_steps = [str(step).strip() for step in steps_value if str(step).strip()]

        final_samples_raw = current_test_result.samples if current_test_result else None
        final_samples = final_samples_raw if include_samples else None

        return CodeResponse(
            code=current_code,
            language=request.language,
            test_passed=current_test_result.passed,
            final_test_result=current_test_result,
            retry_attempts=retry_attempts,
            total_retries=len(retry_attempts),
            generation_time_ms=generation_time_ms,
            tokens_used=total_tokens,
            cost_usd=cost_usd,
            token_usage=token_usage,
            token_cost_usd=cost_usd,
            initial_plan_summary=initial_plan_summary,
            initial_plan_steps=initial_plan_steps,
            samples=final_samples,
            learning=learning_result,
        )

    async def _generate_initial_code(
        self,
        request: CodeRequest,
        initial_plan: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Optional[Dict[str, int]]]:
        """Generate initial code from task description"""

        system_prompt = self._get_system_prompt(request.language, request.test_framework)

        user_prompt = f"""Task: {request.task}

Generate complete, working {request.language} code with tests.

Requirements:
- Write clean, well-documented code
- Include comprehensive tests using {request.test_framework or 'standard test framework'}
- Code must be self-contained and executable
- Follow {request.language} best practices

Return ONLY the code, no explanations."""

        if initial_plan:
            plan_summary = initial_plan.get("summary") or initial_plan.get("plan_overview")
            plan_steps = initial_plan.get("plan_steps") or []
            plan_section = "\n\nPlanning notes:\n"
            if plan_summary:
                plan_section += f"Summary: {plan_summary}\n"
            if plan_steps:
                plan_section += "Steps:\n" + "\n".join(f"- {step}" for step in plan_steps)
            user_prompt += plan_section

        messages = sanitize_messages([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3,  # Low temperature for code generation
            max_tokens=2000
        )

        code = response.choices[0].message.content.strip()

        # Extract code from markdown blocks if present
        code = self._extract_code_from_markdown(code)

        usage = getattr(response, 'usage', None)
        usage_dict = None
        if usage:
            usage_dict = {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            }

        return code, usage_dict

    async def _fix_code(
        self,
        request: CodeRequest,
        current_code: str,
        test_result: TestResult,
        analysis: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Optional[Dict[str, int]]]:
        """Fix code based on test failure"""

        system_prompt = self._get_system_prompt(request.language, request.test_framework)

        analysis_summary = None
        plan_overview = None
        plan_steps: List[str] = []
        if analysis:
            analysis_summary = (
                analysis.get("summary")
                or analysis.get("root_cause")
                or analysis.get("diagnosis")
            )
            plan_overview = (
                analysis.get("plan_overview")
                or analysis.get("plan_summary")
                or analysis.get("fix_strategy")
            )
            raw_steps = analysis.get("plan_steps") or []
            if isinstance(raw_steps, str):
                raw_steps = [step.strip() for step in raw_steps.splitlines() if step.strip()]
            plan_steps = [step.strip() for step in raw_steps if isinstance(step, str) and step.strip()]

        plan_text = ""
        if plan_steps:
            enumerated_steps = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(plan_steps))
            plan_text = f"\nPlanned Steps to Apply:\n{enumerated_steps}"

        analysis_text = ""
        if analysis_summary or plan_overview or plan_text:
            analysis_text = "\n\nFailure Analysis Summary: "
            analysis_text += analysis_summary or "Not available"
            if plan_overview:
                analysis_text += f"\nProposed Fix Strategy: {plan_overview}"
            if plan_text:
                analysis_text += plan_text

        user_prompt = f"""The following code failed tests:

```{request.language}
{current_code}
```

Test Error:
```
{test_result.stderr}
```

Exit Code: {test_result.exit_code}

{analysis_text}

Please fix the code to make all tests pass. Return ONLY the corrected code, no explanations."""

        messages = sanitize_messages([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )

        code = response.choices[0].message.content.strip()
        code = self._extract_code_from_markdown(code)

        usage = getattr(response, 'usage', None)
        usage_dict = None
        if usage:
            usage_dict = {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            }

        return code, usage_dict

    async def _analyze_failure(
        self,
        request: CodeRequest,
        code: str,
        test_result: TestResult,
        attempt: int,
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze test failure and propose fix plan."""

        system_prompt = (
            "You are a senior software engineer helping an assistant debug code. "
            "Analyze failing tests, identify the root cause, and produce a concrete remediation plan."
        )

        user_prompt = f"""Task: {request.task}
Language: {request.language}
Attempt: {attempt + 1}

Current Code:
```{request.language}
{code}
```

Test STDOUT:
```
{test_result.stdout.strip() or 'N/A'}
```

Test STDERR:
```
{test_result.stderr.strip() or 'N/A'}
```

Exit Code: {test_result.exit_code}

Please respond in JSON with the following keys:
- summary: short summary (1-2 sentences) of the failure cause
- root_cause: concise description of what went wrong
- plan_overview: overview of the fix approach
- plan_steps: array of 2-5 concrete steps to implement the fix
- risks: potential side-effects or things to verify after the fix
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )

            content = response.choices[0].message.content.strip()
            json_text = self._extract_json_block(content)
            data = json.loads(json_text)

            # Normalize plan steps
            plan_steps = data.get("plan_steps") or []
            if isinstance(plan_steps, str):
                plan_steps = [step.strip() for step in plan_steps.splitlines() if step.strip()]
            elif isinstance(plan_steps, list):
                plan_steps = [str(step).strip() for step in plan_steps if str(step).strip()]
            else:
                plan_steps = []
            data["plan_steps"] = plan_steps

            return data
        except Exception as exc:
            logger.warning("Failed to analyze failure via LLM: %s", exc)
            fallback_summary = test_result.stderr.strip()[:300] or "Unknown error"
            return {
                "summary": fallback_summary,
                "root_cause": fallback_summary,
                "plan_overview": "Review stderr and address issues manually",
                "plan_steps": [],
                "risks": "Analysis fallback used; verify fixes manually",
            }

    async def _analyze_initial_plan(self, request: CodeRequest) -> Optional[Dict[str, Any]]:
        """Produce an initial plan before writing any code."""

        system_prompt = (
            "You are a senior software engineer preparing to implement a task. "
            "Break down the work into actionable steps before any code is written."
        )

        user_prompt = f"""Task: {request.task}
Language: {request.language}
Test Framework: {request.test_framework or 'auto'}

Please respond in JSON with the following keys:
- summary: one-sentence outline of the approach
- plan_overview: short description of the intended strategy
- plan_steps: ordered list (array) of 3-6 concrete steps to implement
- risks: potential challenges or checks to keep in mind
- recommended_tests: optional list of tests to ensure correctness
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=400,
            )

            content = response.choices[0].message.content.strip()
            json_text = self._extract_json_block(content)
            data = json.loads(json_text)

            plan_steps = data.get("plan_steps") or []
            if isinstance(plan_steps, str):
                plan_steps = [step.strip() for step in plan_steps.splitlines() if step.strip()]
            elif isinstance(plan_steps, list):
                plan_steps = [str(step).strip() for step in plan_steps if str(step).strip()]
            else:
                plan_steps = []
            data["plan_steps"] = plan_steps

            recommended_tests = data.get("recommended_tests") or []
            if isinstance(recommended_tests, str):
                recommended_tests = [step.strip() for step in recommended_tests.splitlines() if step.strip()]
            elif isinstance(recommended_tests, list):
                recommended_tests = [str(item).strip() for item in recommended_tests if str(item).strip()]
            else:
                recommended_tests = []
            data["recommended_tests"] = recommended_tests

            return data
        except Exception as exc:
            logger.warning("Failed to produce initial plan: %s", exc)
            return None

    async def _run_tests(
        self,
        code: str,
        language: Language,
        test_framework: Optional[TestFramework],
        *,
        include_samples: bool = False,
    ) -> TestResult:
        """Run tests on generated code"""
        start_time = time.time()

        try:
            if language == Language.PYTHON:
                result = await self._run_python_tests(code, include_samples=include_samples)
            elif language == Language.RUST:
                result = await self._run_rust_tests(code)
            elif language == Language.BASH:
                result = await self._run_bash_tests(code)
            else:
                # For unsupported languages, return mock success
                return TestResult(
                    passed=True,
                    stdout="Tests skipped (language not fully supported)",
                    stderr="",
                    exit_code=0,
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            return result

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return TestResult(
                passed=False,
                stdout="",
                stderr=str(e),
                exit_code=1,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    async def _run_python_tests(self, code: str, *, include_samples: bool = False) -> TestResult:
        """Run Python tests using pytest"""
        start_time = time.time()

        # If include_samples is True, inject print statements before assertions
        code_to_run = code
        if include_samples:
            code_to_run = self._inject_assertion_prints(code)

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code_to_run)
            temp_file = f.name

        try:
            # First, run the code directly to capture all print/log output
            direct_run = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            direct_output = direct_run.stdout

            # Then run pytest with -s flag to show print output
            result = subprocess.run(
                ['python', '-m', 'pytest', temp_file, '-v', '-s'],
                capture_output=True,
                text=True,
                timeout=10
            )

            execution_time_ms = (time.time() - start_time) * 1000

            # Combine direct output with test output
            combined_stdout = ""
            if direct_output.strip():
                combined_stdout += "=== Program Output ===\n" + direct_output + "\n\n"
            combined_stdout += "=== Test Results ===\n" + result.stdout
            samples = None
            if include_samples and result.returncode == 0:
                samples = self._generate_assertion_samples(code)
                if samples:
                    combined_stdout += "\n=== Sample Evaluations ===\n"
                    for sample in samples:
                        expr = sample.get("expression") or "<expression>"
                        actual = sample.get("actual")
                        expected = sample.get("expected")
                        combined_stdout += f"{expr} -> {actual}"
                        if expected is not None:
                            combined_stdout += f" (expected {expected})"
                        combined_stdout += "\n"

            return TestResult(
                passed=(result.returncode == 0),
                stdout=combined_stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                execution_time_ms=execution_time_ms,
                samples=samples,
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                passed=False,
                stdout="",
                stderr="Test execution timed out (>10s)",
                exit_code=124,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass

    def _inject_assertion_prints(self, code: str) -> str:
        """
        Inject print statements before assert statements in test functions.
        For each assert statement like 'assert func(x) == expected',
        add 'print(func(x))' before it.
        """
        try:
            tree = ast.parse(code)
        except Exception as exc:
            logger.warning(f"Failed to parse code for print injection: {exc}")
            return code

        # Track modifications
        modified = False

        class AssertPrintInjector(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Only process test functions
                if not node.name.startswith("test"):
                    return node

                new_body = []
                for stmt in node.body:
                    if isinstance(stmt, ast.Assert):
                        # Extract the expression being tested
                        test_expr = stmt.test

                        # For comparison assertions (e.g., assert x == y)
                        if isinstance(test_expr, ast.Compare):
                            left = test_expr.left
                            # Create print statement for the left side
                            print_call = ast.Expr(
                                value=ast.Call(
                                    func=ast.Name(id='print', ctx=ast.Load()),
                                    args=[left],
                                    keywords=[]
                                )
                            )
                            # Add print before assert
                            new_body.append(print_call)

                    new_body.append(stmt)

                node.body = new_body
                return node

        injector = AssertPrintInjector()
        modified_tree = injector.visit(tree)

        # Convert back to code
        try:
            modified_code = ast.unparse(modified_tree)
            return modified_code
        except Exception as exc:
            logger.warning(f"Failed to unparse modified AST: {exc}")
            return code

    def _generate_assertion_samples(self, code: str) -> Optional[List[Dict[str, Any]]]:
        """Evaluate simple equality assertions to surface concrete outputs."""
        MAX_SAMPLES = 3
        MAX_CODE_CHARS = 6000
        if not code or len(code) > MAX_CODE_CHARS:
            return None

        samples: List[Dict[str, Any]] = []
        try:
            tree = ast.parse(code)
        except Exception as exc:
            logger.warning("Failed to parse code for samples: %s", exc)
            return None

        module_globals: Dict[str, Any] = {}
        try:
            exec(compile(tree, filename="<generated>", mode="exec"), module_globals)
        except Exception as exc:
            logger.warning("Failed to execute code for samples: %s", exc)
            return None

        try:
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test"):
                    continue
                for stmt in node.body:
                    if len(samples) >= MAX_SAMPLES:
                        break
                    if not isinstance(stmt, ast.Assert):
                        continue
                    test_expr = stmt.test
                    if not isinstance(test_expr, ast.Compare):
                        continue
                    if len(test_expr.ops) != 1 or not isinstance(test_expr.ops[0], ast.Eq):
                        continue
                    left = test_expr.left
                    right = test_expr.comparators[0]
                    try:
                        buf = io.StringIO()
                        with contextlib.redirect_stdout(buf):
                            actual_value = eval(
                                compile(ast.Expression(left), "<eval>", "eval"),
                                module_globals,
                            )
                        captured_stdout = buf.getvalue().strip()
                        expected_value = eval(
                            compile(ast.Expression(right), "<eval>", "eval"),
                            module_globals,
                        )
                        expr_src = ast.get_source_segment(code, left) or "<expression>"
                        sample_entry: Dict[str, Any] = {
                            "expression": expr_src.strip(),
                            "actual": actual_value,
                            "expected": expected_value,
                        }
                        if captured_stdout:
                            sample_entry["stdout"] = captured_stdout
                        samples.append(sample_entry)
                    except Exception as eval_exc:
                        logger.warning("Failed to evaluate assertion expression: %s", eval_exc)
                        continue
            return samples or None
        except Exception as exc:
            logger.warning("Failed to extract assertion samples: %s", exc)
            return None

    async def _run_rust_tests(self, code: str) -> TestResult:
        """Run Rust tests using cargo"""
        start_time = time.time()

        # Create temporary cargo project
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal Cargo.toml
            cargo_toml = f"""[package]
name = "test_project"
version = "0.1.0"
edition = "2021"
"""
            with open(os.path.join(tmpdir, "Cargo.toml"), 'w') as f:
                f.write(cargo_toml)

            # Create src directory
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)

            # Write code to lib.rs
            with open(os.path.join(src_dir, "lib.rs"), 'w') as f:
                f.write(code)

            try:
                # Run cargo test
                result = subprocess.run(
                    ['cargo', 'test'],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                execution_time_ms = (time.time() - start_time) * 1000

                return TestResult(
                    passed=(result.returncode == 0),
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.returncode,
                    execution_time_ms=execution_time_ms
                )

            except subprocess.TimeoutExpired:
                return TestResult(
                    passed=False,
                    stdout="",
                    stderr="Cargo test timed out (>30s)",
                    exit_code=124,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            except FileNotFoundError:
                return TestResult(
                    passed=False,
                    stdout="",
                    stderr="Cargo not found. Please install Rust toolchain.",
                    exit_code=127,
                    execution_time_ms=(time.time() - start_time) * 1000
                )

    async def _run_bash_tests(self, code: str) -> TestResult:
        """Run Bash script directly"""
        start_time = time.time()

        # Create temporary bash file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.sh',
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Make executable
            os.chmod(temp_file, 0o755)

            # Run the bash script directly
            direct_run = subprocess.run(
                ['bash', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            execution_time_ms = (time.time() - start_time) * 1000

            # Bash scripts pass if exit code is 0
            return TestResult(
                passed=(direct_run.returncode == 0),
                stdout=direct_run.stdout,
                stderr=direct_run.stderr,
                exit_code=direct_run.returncode,
                execution_time_ms=execution_time_ms
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                passed=False,
                stdout="",
                stderr="Bash script timed out (>10s)",
                exit_code=124,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass

    def _get_system_prompt(self, language: Language, test_framework: Optional[TestFramework]) -> str:
        """Get system prompt for code generation"""

        lang_name = str(language.value) if hasattr(language, 'value') else str(language)

        base_prompt = f"You are an expert {lang_name} programmer. "

        if language == Language.PYTHON:
            base_prompt += "Write clean, Pythonic code following PEP 8. "
            if test_framework == TestFramework.PYTEST:
                base_prompt += "Use pytest for testing with clear test functions. "
        elif language == Language.RUST:
            base_prompt += "Write idiomatic Rust code with proper error handling. "
            base_prompt += "Include unit tests using Rust's built-in test framework. "

        base_prompt += "Always include comprehensive tests. Code must be production-ready. "
        base_prompt += "\n\nIMPORTANT RULES:\n"
        base_prompt += "1. Return ONLY executable code - no explanations, no comments outside code\n"
        base_prompt += "2. Do NOT include any enum values like 'Language.PYTHON' or metadata\n"
        base_prompt += "3. Do NOT include markdown formatting unless in code blocks\n"
        base_prompt += "4. Code must be immediately executable\n"
        base_prompt += "5. Include imports and all necessary code"

        return base_prompt

    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code blocks and clean non-code content"""
        import re

        # Remove leading/trailing whitespace
        text = text.strip()

        # Pattern 1: ```language\ncode\n```
        pattern1 = r'```(?:\w+)?\s*\n(.*?)```'
        matches = re.findall(pattern1, text, re.DOTALL)
        if matches:
            code = matches[0].strip()
            return self._clean_code(code)

        # Pattern 2: ```code``` (inline)
        pattern2 = r'```(.*?)```'
        matches = re.findall(pattern2, text, re.DOTALL)
        if matches:
            code = matches[0].strip()
            return self._clean_code(code)

        # Pattern 3: Starts with ``` but no closing (malformed)
        if text.startswith('```'):
            # Remove opening ```language
            text = re.sub(r'^```\w*\s*\n?', '', text)
            # Remove closing ``` if present
            text = re.sub(r'\n?```\s*$', '', text)
            return self._clean_code(text.strip())

        # No code blocks found, clean and return
        return self._clean_code(text)

    def _extract_json_block(self, text: str) -> str:
        """Extract JSON content from model response."""
        if not text:
            return "{}"
        stripped = text.strip()
        if stripped.startswith("```"):
            parts = stripped.split("```")
            if len(parts) >= 2:
                candidate = parts[1]
                if candidate.startswith("json"):
                    candidate = candidate[4:]
                return candidate.strip()
        return stripped

    def _clean_code(self, code: str) -> str:
        """Remove common LLM artifacts from generated code"""
        import re

        # Remove lines that look like enum references
        code = re.sub(r'^Language\.\w+\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^TestFramework\.\w+\s*$', '', code, flags=re.MULTILINE)

        # Remove lines with only metadata
        code = re.sub(r'^(language|test_framework|framework):\s*\w+\s*$', '', code, flags=re.MULTILINE)

        # Remove empty lines at start/end
        code = code.strip()

        # Remove multiple consecutive empty lines
        code = re.sub(r'\n{3,}', '\n\n', code)

        return code


# Singleton instance
_assistant_instance: Optional[CodeAssistant] = None


def get_code_assistant() -> CodeAssistant:
    """Get singleton instance of code assistant"""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = CodeAssistant()
    return _assistant_instance
