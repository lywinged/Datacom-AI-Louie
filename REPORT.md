# Technical Report: AI Assessment Project

**Date:** October 2025
## Executive Summary

This report outlines key design decisions and trade-offs in building an enterprise AI platform with Chat, RAG, Agent, and Code Assistant capabilities. The system achieves 95% API test coverage with sub-second response times while maintaining production-ready reliability.

# Highlight

1.Fully satisfies all detailed requirements of the four tasks.

2.The four core services — AI Assistant, RAG, Trip Agent, and Code Generation — can switch seamlessly based on natural language, with the LLM serving as intent recognition.

3.RAG supports optional models and rerankers like bge-m3, automatically balancing trade-offs between model size, speed, and GPU availability. After selecting a model, users can adjust vector limits and vector sizes to fine-tune the latency–accuracy trade-off of the reranker.

4.The Trip Agent collects departure location, destination, duration, and budget in the first round. It includes an extreme-request rejection mechanism to save resources and improve response speed(like I plan to go New York from Auckland with $1, we won't call LLM or Function call for this request).

It also automatically retrieves exchange rates via API based on the departure location to convert currency units for better user experience. 

I add the industry-leading feature of “context adaptation” — unsupervised online learning that improves the one-shot satisfaction rate of trip plans through continuous user interactions while saving resources.(As above example, consider a round trip from Beijing to Auckland with a budget of only $200, while it’s technically greater than $1, it’s still far beyond common sense. so how to automatically optimizes trip planning strategies are precisely the kind of case we want the model to learn from.)
*Please refer docs/TRIP_LEARNING.md

5.Code Generation is designed with the same unsupervised online learning capability, supporting automatic multi-language detection. It comes with a preinstalled Python environment for quick testing and displaying execution outputs.（The purpose of my online reinforcement learning task is to reuse previously failed coding plans and summarize the optimal coding strategies across different tasks.） 
*Please refer docs/CODE_LEARNING.md

**Key Metrics:**
- 📊 29/29 tests passing (0.54s execution)
- 🚀 RAG queries: Median 209ms RAG part
- 💬 Chat responses: ~500ms (GPT-4)
- 📈 95% API endpoint coverage

The architecture is production-ready and validated through automated testing, performance benchmarks, and Docker deployment.

---

