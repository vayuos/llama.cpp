# Contributors

This project operates under a fully decentralized contribution model.

There are no formal roles, tiers, or privileged classes of contributors. Anyone may:

* Propose changes
* Review pull requests
* Suggest improvements
* Participate in discussions

Reputation and influence are earned organically through consistent, high-quality contributions.

All participants are equally responsible for maintaining code quality and project integrity.

---

# AI Usage Policy

AI usage is permitted.

However:

* Every contributor is fully responsible for the correctness, safety, and performance of the code they submit.
* Contributors must manually review all code prior to submission.
* Contributors must be able to explain every line of submitted code.
* Low-effort or unverified AI-generated contributions may be rejected by community consensus.

AI may be used for:

* Drafting implementations
* Refactoring
* Generating boilerplate
* Improving documentation

AI must not be used as a substitute for understanding the system.

---

# Pull Requests

Anyone may submit and review pull requests.

Before submitting:

* Search for existing issues and PRs to avoid duplication.
* Test changes locally.
* Ensure performance, correctness, and compatibility are not negatively impacted.
* Keep each PR focused on a single logical change.

Strongly encouraged:

* Run CI locally where possible.
* Verify perplexity and performance when relevant.
* Follow existing coding and naming conventions.
* Avoid unrelated changes within the same PR.

After submission:

* Acceptance is determined through open community discussion.
* Any contributor may provide review feedback.
* Consensus drives merging decisions.
* Stale PRs should be rebased on the latest `master`.

---

# Merging Process

There are no permanent maintainers.

Merging authority is granted through repository configuration (e.g., write access). Those with merge permissions:

* Must act in the interest of project stability.
* Must not merge unreviewed or controversial changes.
* May revert breaking or harmful changes when necessary.

Merge decisions should reflect visible community agreement.

Recommended squash-merge commit title format:

```
<module> : <commit title> (#<issue_number>)
```

---

# Coding Guidelines

Guidelines are strongly encouraged to maintain coherence:

* Avoid unnecessary dependencies.
* Preserve cross-platform compatibility.
* Keep implementations simple and readable.
* Follow existing indentation and formatting style.
* Use 4 spaces for indentation.
* Avoid trailing whitespace.
* Use `snake_case` for functions, variables, and types.
* Prefer sized integer types (`int32_t`, etc.) in public APIs.
* Follow established naming patterns (`<class>_<action>_<noun>`).

Existing patterns in the codebase take precedence over personal stylistic preferences.

---

# Code Maintenance

All contributors share responsibility for maintenance.

When modifying substantial portions of code:

* Be prepared to support and fix related issues.
* Provide tests where appropriate.
* Ensure CI compatibility.

Large architectural changes should be discussed publicly before implementation.

Any contributor may propose structural or governance changes through issues or pull requests.

---

# Documentation

Documentation is community-maintained.

Contributors are encouraged to:

* Update outdated documentation.
* Clarify API usage in header files.
* Improve developer onboarding material.
* Add examples where helpful.

Accuracy and clarity are collective responsibilities.

---

# Governance Model

This project follows a fully decentralized governance philosophy:

* No permanent maintainers.
* No privileged contributor tiers.
* Open review by any participant.
* Decisions guided by visible consensus.
* Repository integrity maintained through shared responsibility.
* Reversions allowed when required to preserve stability.

Authority emerges from contribution quality, not assigned roles.
