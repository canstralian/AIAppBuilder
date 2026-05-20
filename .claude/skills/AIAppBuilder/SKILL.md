```markdown
# AIAppBuilder Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the AIAppBuilder repository, a TypeScript codebase designed without a specific framework. You'll learn how to structure files, write imports/exports, follow commit message styles, and implement or run tests. This guide also provides command suggestions for common workflows.

## Coding Conventions

### File Naming
- Use **camelCase** for all file names.
  - Example: `userProfile.ts`, `dataManager.test.ts`

### Import Style
- Use **relative imports** for referencing modules within the project.
  - Example:
    ```typescript
    import { fetchData } from './apiUtils';
    ```

### Export Style
- Use **named exports** for all modules.
  - Example:
    ```typescript
    // In userProfile.ts
    export function getUserProfile(id: string) { ... }

    // In another file
    import { getUserProfile } from './userProfile';
    ```

### Commit Messages
- Commit messages are **freeform** (no strict prefix), with an average length of 52 characters.
  - Example:  
    ```
    Add user authentication logic and update API calls
    ```

## Workflows

### Adding a New Feature
**Trigger:** When implementing a new functionality.
**Command:** `/add-feature`

1. Create a new file using camelCase naming (e.g., `featureName.ts`).
2. Write your TypeScript code, using relative imports and named exports.
3. Add or update relevant test files (e.g., `featureName.test.ts`).
4. Commit your changes with a clear, descriptive message.
5. Push your branch and open a pull request.

### Running Tests
**Trigger:** When verifying code correctness.
**Command:** `/run-tests`

1. Locate test files matching the `*.test.*` pattern.
2. Use the project's test runner (framework unknown; check `package.json` or documentation).
3. Run all tests and review results.
4. Fix any failing tests before merging changes.

### Refactoring Code
**Trigger:** When improving code structure or readability.
**Command:** `/refactor`

1. Identify files or modules to refactor.
2. Update code, maintaining camelCase file naming, relative imports, and named exports.
3. Update or add tests as needed.
4. Commit changes with a descriptive message.
5. Run tests to ensure nothing is broken.

## Testing Patterns

- Test files follow the pattern: `*.test.*` (e.g., `apiUtils.test.ts`).
- The specific testing framework is unknown; check project documentation or `package.json` for details.
- Place tests alongside or near the code they cover.
- Example test file:
  ```typescript
  // dataManager.test.ts
  import { fetchData } from './dataManager';

  test('fetchData returns correct result', () => {
    expect(fetchData()).toBeDefined();
  });
  ```

## Commands
| Command        | Purpose                                    |
|----------------|--------------------------------------------|
| /add-feature   | Start the workflow for adding a new feature|
| /run-tests     | Run all tests in the codebase              |
| /refactor      | Begin a code refactoring workflow          |
```
