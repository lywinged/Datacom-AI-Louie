#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive Code Generation Chat"""

import requests
import json
import sys


class CodeChatClient:
    def __init__(self, base_url="http://localhost:8888"):
        self.base_url = base_url
        self.code_endpoint = f"{base_url}/api/code/generate"
        self.health_endpoint = f"{base_url}/api/code/health"

    def check_health(self):
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Cannot connect to Code API: {e}")
            return False

    def generate_code(self, task, language="python", max_retries=3):
        """Generate code with pseudo-streaming progress display"""
        import threading
        import time

        try:
            payload = {
                "task": task,
                "language": language,
                "max_retries": max_retries
            }

            # Start progress animation in background
            stop_animation = threading.Event()
            animation_thread = threading.Thread(
                target=self._animate_progress,
                args=(stop_animation, max_retries)
            )
            animation_thread.start()

            # Make the actual request
            response = requests.post(
                self.code_endpoint,
                json=payload,
                timeout=120
            )

            # Stop animation
            stop_animation.set()
            animation_thread.join()

            if response.status_code == 200:
                return response.json()
            else:
                print(f"\n❌ Error: {response.status_code}")
                print(response.text)
                return None

        except Exception as e:
            if 'stop_animation' in locals():
                stop_animation.set()
            print(f"\n❌ Request failed: {e}")
            return None

    def _animate_progress(self, stop_event, max_retries):
        """Animate progress while waiting for response"""
        import time

        stages = [
            ("🔨 Generating initial code", 2),
            ("⏳ Waiting for LLM response", 8),
            ("✅ Code generated", 1),
            ("🧪 Running tests", 2),
            ("⏳ Executing test framework", 3),
        ]

        stage_idx = 0
        elapsed = 0

        print()  # New line

        while not stop_event.is_set():
            if stage_idx < len(stages):
                stage_text, duration = stages[stage_idx]

                # Print stage
                # print(f"\r{stage_text}{'.' * (elapsed % 4)}", end='', flush=True)
                print(f"\r{stage_text}{'.' * (int(elapsed) % 4)}", end='', flush=True)
                # 或者更稳妥：n 点在 0..3 之间循环
                n = int(elapsed) % 4
                print(f"\r{stage_text}{'.' * n}", end='', flush=True)

                time.sleep(0.5)
                elapsed += 0.5

                # Move to next stage
                if elapsed >= duration:
                    print(f"\r{stage_text}... Done!")
                    stage_idx += 1
                    elapsed = 0

                    # Add retry stages dynamically
                    if stage_idx == len(stages):
                        for retry_num in range(1, max_retries + 1):
                            if not stop_event.is_set():
                                print(f"❌ Tests failed")
                                print(f"🔧 Retry {retry_num}/{max_retries} - Fixing code...")
                                time.sleep(1)
                                if stop_event.is_set():
                                    break
                                print(f"⏳ Waiting for LLM response...")
                                time.sleep(2)
                                if stop_event.is_set():
                                    break
                                print(f"✅ Code fixed")
                                print(f"🧪 Running tests again...")
                                time.sleep(1.5)
                                if stop_event.is_set():
                                    break
                        break
            else:
                time.sleep(0.2)

        print()  # Final newline

    def display_result(self, result):
        if not result:
            return

        # Print final status first (continues the streaming output)
        if result.get("test_passed"):
            print("✅ All tests passed!")
        else:
            print("❌ Tests failed after max retries")

        print("\n" + "=" * 100)
        print("CODE GENERATION RESULT")
        print("=" * 100 + "\n")

        # Show if tests passed
        status = "PASSED" if result.get("test_passed") else "FAILED"
        status_emoji = "✅" if result.get("test_passed") else "❌"

        print(f"Status: {status_emoji} {status}")
        print(f"Language: {result.get('language')}")
        print(f"Retries: {result.get('total_retries')}/{result.get('max_retries', 3)}")
        print(f"Time: {result.get('generation_time_ms', 0):.0f}ms")
        print(f"Tokens: {result.get('tokens_used', 0)}")
        print(f"Cost: ${result.get('cost_usd', 0):.4f}")
        print()
        
        # Show generated code
        print("-" * 100)
        print("GENERATED CODE")
        print("-" * 100)
        print(result.get("code", ""))
        print()
        
        # Show execution output
        final_test = result.get("final_test_result", {})
        print("-" * 100)
        print("EXECUTION OUTPUT")
        print("-" * 100)
        print(f"Exit Code: {final_test.get('exit_code', 'N/A')}")
        print(f"Execution Time: {final_test.get('execution_time_ms', 0):.0f}ms")
        print()

        # Show stdout (includes both print statements and test output)
        stdout = final_test.get("stdout", "")
        if stdout:
            print("Program Output:")
            print(stdout)
            print()

        # Show stderr (errors and warnings)
        stderr = final_test.get("stderr", "")
        if stderr:
            print("Errors/Warnings:")
            print(stderr)
            print()
        
        # Show retry history
        retries = result.get("retry_attempts", [])
        if retries:
            print("\n" + "-" * 100)
            print("RETRY HISTORY")
            print("-" * 100)
            for retry in retries:
                print(f"\nAttempt {retry.get('attempt_number')}:")
                print(f"  Fix Applied: {retry.get('fix_applied', 'N/A')}")
                root_cause = retry.get("error_analysis")
                if root_cause:
                    print(f"  Root Cause: {root_cause}")
                plan_summary = retry.get("plan_summary")
                if plan_summary:
                    print(f"  Plan Overview: {plan_summary}")
                plan_steps = retry.get("plan_steps") or []
                if plan_steps:
                    print("  Plan Steps:")
                    for idx, step in enumerate(plan_steps, start=1):
                        print(f"    {idx}. {step}")

        print("\n" + "=" * 100)

    def interactive_chat(self):
        print("=" * 100)
        print("Code Generation Assistant")
        print("=" * 100 + "\n")
        print("Commands:")
        print("  - Type your coding task")
        print("  - 'lang <language>' to change language (python, rust, javascript, go)")
        print("  - 'retries <N>' to set max retries (1-5)")
        print("  - 'quit' or 'exit' to exit")
        print("=" * 100 + "\n")
        
        if not self.check_health():
            print("\nCode API not running!")
            print("Start server: python -m uvicorn backend.main:app --port 8888\n")
            return
        
        print("Code API connected!\n")
        
        language = "python"
        max_retries = 3
        request_count = 0
        
        while True:
            try:
                print("-" * 100)
                print(f"\nCurrent settings: language={language}, max_retries={max_retries}")
                user_input = input("\nYour task: ").strip()
                print()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower().startswith('lang '):
                    new_lang = user_input.split(' ', 1)[1]
                    if new_lang in ['python', 'rust', 'javascript', 'go']:
                        language = new_lang
                        print(f"Language set to: {language}")
                    else:
                        print("Supported languages: python, rust, javascript, go")
                    continue
                
                elif user_input.lower().startswith('retries '):
                    try:
                        new_retries = int(user_input.split(' ', 1)[1])
                        if 1 <= new_retries <= 5:
                            max_retries = new_retries
                            print(f"Max retries set to: {max_retries}")
                        else:
                            print("Retries must be between 1 and 5")
                    except ValueError:
                        print("Invalid number")
                    continue
                
                # Generate code
                result = self.generate_code(user_input, language, max_retries)
                
                if result:
                    request_count += 1
                    self.display_result(result)
                else:
                    print("Generation failed")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Code Generation")
    parser.add_argument("--url", default="http://localhost:8888", help="API URL")
    parser.add_argument("--task", help="Single task (non-interactive)")
    parser.add_argument("--language", default="python", help="Programming language")
    parser.add_argument("--retries", type=int, default=3, help="Max retries")
    args = parser.parse_args()
    
    client = CodeChatClient(base_url=args.url)
    
    if args.task:
        print(f"Task: {args.task}\n")
        result = client.generate_code(args.task, args.language, args.retries)
        if result:
            client.display_result(result)
        else:
            sys.exit(1)
    else:
        client.interactive_chat()


if __name__ == "__main__":
    main()
