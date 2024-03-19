module.exports = {
    // ...
    extends: ["plugin:svelte/recommended", "standard-with-typescript"],
    // ...
    parser: '@typescript-eslint/parser',
    parserOptions: {
        // ...
        project: 'tsconfig.json',
        extraFileExtensions: ['.svelte'] // This is a required setting in `@typescript-eslint/parser` v4.24.0.
    },
    overrides: [
        {
            files: ['*.svelte'],
            parser: 'svelte-eslint-parser',
            // Parse the `<script>` in `.svelte` as TypeScript by adding the following configuration.
            parserOptions: {
                parser: '@typescript-eslint/parser'
            }
        },
        {
            rules: {
                "@typescript-eslint/naming-convention": [
                    "warn",
                    {
                        "selector": "variable",
                        "format": ["camelCase", "UPPER_CASE", "snake_case"]
                    }
                ]
            }
        }
    ]
}