export default {
    ignores: [
        (message) => message.startsWith("Use escalation span provenance for memory ingest"),
        (message) => message.startsWith("Bump uipath-langchain version to 0.10.24"),
    ],
    rules: {
        "body-max-line-length": [2, "always", 100],
        "footer-max-line-length": [2, "always", 100],
        "header-max-length": [2, "always", 100],
        "subject-empty": [2, "never"],
        "type-empty": [2, "never"],
        "type-enum": [
            2,
            "always",
            ["build", "chore", "ci", "docs", "feat", "fix", "perf", "refactor", "revert", "style", "test"],
        ],
    },
};
