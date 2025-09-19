---
argument-hint: [python_file]
description: Refactor Python code to follow professional best practices and design patterns
allowed-tools: Read, Edit, MultiEdit, Write, Bash, Bash(python:*)
---

Transform the Python file $1 into structured, maintainable code following industry best practices, SOLID principles, and Gang of Four design patterns. Add a module level comment saying that it is created to support
agentic ai execution.

**IMPORTANT**: Only refactor the provided Python file. Do NOT create additional files unless absolutely necessary. Preserve all existing functionality while dramatically improving code quality.

Apply the following refactoring principles to $1:

## 1. Code Organization and Structure

### Constants and Configuration
- Extract all magic strings and numbers into named constants at module level
- Group related constants using Enum classes where appropriate
- Compile regex patterns at module level for performance (use `re.compile()`)
- Use typing hints throughout for better code documentation

### Class Design (Single Responsibility Principle)
- Break large classes into smaller, focused classes with single responsibilities
- Apply composition over inheritance where appropriate
- Use dependency injection for better testability
- Implement proper encapsulation with private methods and properties

## 2. Method Decomposition and Clarity

### Function Decomposition
- Break methods longer than 20 lines into smaller, focused functions
- Each function should do one thing well (Single Responsibility)
- Use descriptive function and variable names that explain intent
- Eliminate complex nested loops and conditional chains

### Control Flow Optimization
- Replace long if/elif/else chains with Strategy pattern or dispatch tables
- Use early returns to reduce nesting levels
- Apply Guard Clauses to handle edge cases upfront
- Convert complex loops into comprehensions or generator expressions where appropriate

## 3. Error Handling and Logging

### Exception Management
- Use specific exception types instead of generic Exception
- Implement proper error hierarchies for different error categories
- Add contextual error messages with relevant details
- Use try-except blocks judiciously, not as control flow

### Logging Integration
- Replace print statements with proper logging
- Use different log levels appropriately (DEBUG, INFO, WARNING, ERROR)
- Add structured logging with context information

## 4. Performance Optimizations

### Regex and String Operations
- Compile all regex patterns at module level
- Use string formatting methods consistently (prefer f-strings for simple cases)
- Avoid string concatenation in loops


### Data Structure Optimization
- Use appropriate data structures (sets for membership testing, deques for queue operations)
- Implement generators for memory-efficient processing
- Cache expensive computations where appropriate

## 5. Type Safety and Documentation

### Type Annotations
- Add comprehensive type hints to all functions and methods
- Use generic types and protocol classes where appropriate
- Leverage mypy-compatible annotations

### Documentation
- Add comprehensive docstrings following Google style
- Include type information, parameter descriptions, and return values


## Implementation Strategy

### Phase 1: Structure Analysis
1. **Identify responsibilities** - Map each method to its core responsibility
2. **Find patterns** - Look for repeated logic that can be extracted
3. **Detect code smells** - Long methods, deep nesting, magic numbers
4. **Decomposition** - Design new class and method structure
5. **Constant Extraction and Regex Compilation**

### Phase 2: Method Simplification
- Extract complex loops into separate methods
- Use guard clauses to reduce nesting
- Apply early returns for cleaner control flow
- Convert nested conditionals to strategy objects

### Phase 3: Integration and Testing
- Ensure all functionality is preserved
- Add comprehensive error handling
- Test edge cases and error conditions

## Validation Checklist

**Structure**: Classes follow Single Responsibility Principle
**Constants**: All magic strings/numbers extracted to constants
**Regex**: All patterns compiled at module level
**Types**: Complete type annotations throughout
**Errors**: Specific exception types with proper handling
**Logging**: Structured logging replaces print statements
**Patterns**: Appropriate GoF patterns applied where beneficial
**Performance**: Optimized for speed and memory usage
**Documentation**: Comprehensive docstrings and comments
**Data Classes**: Use @dataclass for structured data containers
**Enums**: Use Enum classes for related constants and options
**Naming**: Function and method names are meaningful and descriptive
**Lazy Loading**: Use lazy loading for expensive string operations and resources

Transform the target file into a professional, maintainable codebase that exemplifies Python best practices and modern software engineering principles.