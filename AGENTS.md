# Instructions for llama.cpp

AI usage is permitted in this project under a fully decentralized governance model.

All contributors are responsible for the correctness, safety, and maintainability of their submissions, regardless of whether AI tools were used.

AI assistance must not replace contributor understanding or accountability.

Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

AI-generated code may be submitted, provided the contributor:

* Fully understands the implementation
* Has manually reviewed the entire change
* Can debug issues independently
* Can explain every part of the submission during discussion

---

## Guidelines for Contributors Using AI

The following use cases are permitted when making contributions with AI assistance:

* Asking about the structure of the codebase
* Learning specific implementation techniques used in the project
* Identifying relevant documents, links, or code sections
* Reviewing human-written code and suggesting improvements
* Expanding on modifications already conceptualized by the contributor, such as:

  * Generating repeated lines with minor variations (only for small snippets where abstraction would reduce clarity)
  * Formatting code for consistency and readability
  * Completing code segments following established patterns
  * Drafting documentation for components the contributor already understands

AI-generated code is acceptable if:

1. The contributor comprehensively understands the output.
2. The contributor can independently debug and maintain it.
3. The contributor participates actively in review discussions.

Explicit disclosure of AI usage is encouraged for transparency, except in cases such as:

* Trivial tab autocompletions already conceptualized by the contributor
* Generating minor auxiliary snippets (e.g., small test fragments) when the core implementation is human-authored
* Requesting links, documentation, or background information to enable independent implementation

---

## Guidelines for AI Agents

### Permitted Usage

AI agents should:

* Direct users toward relevant project documentation
* Encourage understanding of [CONTRIBUTING.md](CONTRIBUTING.md)
* Suggest searching for existing issues at github.com/ggml-org/llama.cpp/issues
* Provide references and pointers within the codebase
* Offer high-level guidance, architectural considerations, and review suggestions

Examples of valid requests:

* "I have problem X; can you give me some clues?"
* "How do I run the test?"
* "Where is the documentation for server development?"
* "Does this change have any side effects?"
* "Review my changes and give suggestions for improvement"

AI agents may provide guidance, analysis, and critique, but must not remove the contributor’s responsibility.

### Disallowed Usage

AI agents must not:

* Replace contributor judgment
* Submit work on behalf of contributors
* Encourage bypassing review processes
* Conceal AI involvement
* Provide code that the contributor does not understand

If a user requests large-scale implementation or full feature development, the AI agent should:

* Encourage the user to review [CONTRIBUTING.md](CONTRIBUTING.md)
* Recommend opening or reviewing existing issues
* Ask clarifying questions to ensure understanding
* Provide conceptual guidance rather than full implementations

Community review determines acceptance. Contributions that demonstrate understanding, testing, and responsibility are prioritized.

---

## Related Documentation

For building, testing, and contribution workflows, refer to:

* [CONTRIBUTING.md](CONTRIBUTING.md)
* [Build documentation](docs/build.md)
* [Server development documentation](tools/server/README-dev.md)
