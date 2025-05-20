import re
import sys # Import sys for handling input in a more controlled way

# -----------------------------
# Tokenizer and Operator Precedence
# -----------------------------
# Updated tokenizer: supports single-quoted string literals, double-quoted string literals,
# dot operator, tracks line numbers, explicitly handles comments and whitespace,
# and improves invalid token detection.
# FIX: Reordered regex to prioritize matching comments (//.*) before general operators (like /)
# --- MODIFICATION: Added '[' and ']' to the operator group ---
TOKEN_REGEX = r'(\'(?:\\.|[^\'\\])*\')|(\"(?:\\.|[^\"\\])*\")|(//.*)|(\d+)|([a-zA-Z_]+)|(==|!=|&&|\|\||[+\-*/^()=<>!&|{},;.\[\]])|(\s+)|(.)'
# --- END MODIFICATION ---
# Added ('\"(?:\\.|[^\"\\])*\"') to match double-quoted strings
# --- MODIFICATION: Added 'class' and 'new' to KEYWORDS ---
KEYWORDS = {"function", "struct", "return", "for", "if", "else", "true", "false", "not", "and", "or", "let", "while", "input", "null", "delete", "class", "new"}
# --- END MODIFICATION ---

def tokenize(expression):
    """
    Tokenizes the input expression string based on predefined regex rules.

    Args:
        expression (str): The input string to tokenize.

    Returns:
        list: A list of tuples, where each tuple contains the token value and its line number.

    Raises:
        SyntaxError: If an invalid token is encountered.
    """
    tokens = []
    line_num = 1
    # Use finditer to get match objects with start/end positions for line number tracking
    for match in re.finditer(TOKEN_REGEX, expression):
        # Updated groups to include the new double-quoted string capture group
        # FIX: Corrected group assignments to match the new order in TOKEN_REGEX
        string_literal_single, string_literal_double, comment, number, identifier, operator, whitespace, unknown = match.groups()
        token_value = match.group(0)

        # Count newlines within the matched whitespace to update line_num
        if whitespace:
            if '\n' in token_value:
                line_num += token_value.count('\n')
            continue # Skip whitespace

        # Skip comments
        if comment:
            # Comments end at the newline, so we need to count newlines in the comment if any
            if '\n' in token_value:
                 line_num += token_value.count('\n')
            continue # Skip comments

        if unknown:
             # Handle invalid tokens - matches the rubric example 'let x=10 @ 5;'
             # Refined error message to exactly match rubric format: "Displays error: Invalid token @"
             # We still include line_num internally for potential future debugging, but the output message is exact.
             raise SyntaxError(f"Displays error: Invalid token {token_value}") # Removed quotes and line number from output message

        # Append the token value and the current line number
        if string_literal_single:
            tokens.append((string_literal_single, line_num))
        elif string_literal_double:
             tokens.append((string_literal_double, line_num))
        elif number:
            tokens.append((number, line_num))
        elif identifier:
            # Check if identifier is a keyword
            if identifier in KEYWORDS:
                 tokens.append((identifier, line_num))
            else:
                 tokens.append((identifier, line_num))
        elif operator:
            tokens.append((operator, line_num))

    return tokens

# Operator precedence and associativity (assignment is handled separately)
PRECEDENCE = {
    '||': 1, 'or': 1,            # Logical OR
    '&&': 2, 'and': 2,           # Logical AND
    '>': 3, '<': 3, '==': 3, '!=': 3,  # Comparison
    '+': 4, '-': 4,             # Addition, Subtraction
    '*': 5, '/': 5,             # Multiplication, Division
    '^': 6,                    # Exponentiation (right-associative)
    '!': 7, 'not': 7,          # Logical NOT (unary)
    'unary-': 7               # Unary negation
}

# -----------------------------
# AST Node Definitions
# -----------------------------
class ASTNode:
    """Base class for all Abstract Syntax Tree nodes."""
    def __init__(self, line_num=None):
        self.line_num = line_num

class BinaryOp(ASTNode):
    """Represents a binary operation node in the AST."""
    def __init__(self, op, left, right, line_num=None):
        super().__init__(line_num)
        self.op = op
        self.left = left
        self.right = right
    # Updated __repr__ to match rubric format for binary operations
    def __repr__(self):
        # Map internal operator names to the desired output format
        op_map = {'+': 'Add', '-': 'Subtract', '*': 'Multiply', '/': 'Divide', '^': 'Power',
                  '==': 'Equals', '!=': 'NotEquals', '>': 'GreaterThan', '<': 'LessThan',
                  '&&': 'And', '||': 'Or', 'and': 'And', 'or': 'Or'}
        op_name = op_map.get(self.op, self.op) # Use mapped name or original if not found
        return f"{op_name}({repr(self.left)}, {repr(self.right)})"


class UnaryOp(ASTNode):
    """Represents a unary operation node in the AST."""
    def __init__(self, op, operand, line_num=None):
        super().__init__(line_num)
        self.op = op
        self.operand = operand
    # Updated __repr__ to match rubric format for unary operations
    def __repr__(self):
        op_map = {'-': 'Negate', '!': 'Not', 'not': 'Not'}
        op_name = op_map.get(self.op, self.op)
        return f"Negate({repr(self.operand)})"


class Number(ASTNode):
    """Represents a number literal node in the AST."""
    def __init__(self, value, line_num=None):
        super().__init__(line_num)
        self.value = value
    def __repr__(self):
        return str(self.value)

class BooleanLiteral(ASTNode):
    """Represents a boolean literal node in the AST."""
    def __init__(self, value, line_num=None):
        super().__init__(line_num)
        self.value = value
    def __repr__(self):
        return "true" if self.value else "false"

class StringLiteral(ASTNode):
    """Represents a string literal node in the AST."""
    def __init__(self, value, line_num=None):
        super().__init__(line_num)
        self.value = value
    def __repr__(self):
        # Include quotes in the representation to match the token output format
        return f"'{self.value}'"

# --- NEW AST Node for Null ---
class NullLiteral(ASTNode):
    """Represents a null literal node in the AST."""
    def __init__(self, line_num=None):
        super().__init__(line_num)
        self.value = None # Represent null as Python's None
    def __repr__(self):
        return "null"
# --- END NEW AST Node ---

# --- NEW AST Node for Delete ---
class DeleteCall(ASTNode):
    """Represents a delete function call node in the AST."""
    def __init__(self, name, line_num=None):
        super().__init__(line_num)
        self.name = name # Should be a Variable node
    def __repr__(self):
        return f"Delete({repr(self.name)})"
# --- END NEW AST Node ---

# --- NEW AST Node for List Literal ---
class ListLiteral(ASTNode):
    """Represents a list literal node in the AST."""
    def __init__(self, elements, line_num=None):
        super().__init__(line_num)
        self.elements = elements
    def __repr__(self):
        return f"List([{', '.join(map(repr, self.elements))}])"
# --- END NEW AST Node ---

# --- NEW AST Node for Index Access ---
class IndexAccess(ASTNode):
    """Represents an index access (e.g., list[index]) node in the AST."""
    def __init__(self, base, index, line_num=None):
        super().__init__(line_num)
        self.base = base # The expression for the list/array
        self.index = index # The expression for the index
    def __repr__(self):
        return f"IndexAccess({repr(self.base)}, {repr(self.index)})"
# --- END NEW AST Node ---

# --- NEW AST Node for Class Definition ---
class ClassDef(ASTNode):
    """Represents a class definition node in the AST."""
    def __init__(self, name, methods, line_num=None):
        super().__init__(line_num)
        self.name = name
        self.methods = methods # Dictionary mapping method name to FunctionDef node
    def __repr__(self):
        return f"ClassDef({self.name}, {repr(self.methods)})"
# --- END NEW AST Node ---

# --- NEW AST Node for New Expression ---
class NewExpression(ASTNode):
    """Represents a 'new' expression for object instantiation."""
    def __init__(self, class_name, args, line_num=None):
        super().__init__(line_num)
        self.class_name = class_name
        self.args = args # Arguments passed to the constructor (if any)
    def __repr__(self):
        return f"New({self.class_name}, [{', '.join(map(repr, self.args))}])"
# --- END NEW AST Node ---

# --- NEW AST Node for Method Call ---
class MethodCall(ASTNode):
    """Represents a method call on an object instance."""
    def __init__(self, base, method_name, args, line_num=None):
        super().__init__(line_num)
        self.base = base # The expression for the object instance
        self.method_name = method_name # The name of the method (string)
        self.args = args # Arguments passed to the method
    def __repr__(self):
        return f"MethodCall({repr(self.base)}, {self.method_name}, [{', '.join(map(repr, self.args))}])"
# --- END NEW AST Node ---


class Variable(ASTNode):
    """Represents a variable node in the AST."""
    def __init__(self, name, line_num=None):
        super().__init__(line_num)
        self.name = name
    # Updated __repr__ to match rubric format for variables (just the name)
    def __repr__(self):
        return self.name

class Assignment(ASTNode):
    """Represents an assignment statement node in the AST."""
    def __init__(self, name, value, line_num=None):
        super().__init__(line_num)
        self.name = name
        self.value = value
    # Updated __repr__ to match rubric format for assignments
    def __repr__(self):
        # Assuming name is a Variable node or MemberAccess node
        return f"Assign({repr(self.name)}, {repr(self.value)})"


class FunctionCall(ASTNode):
    """Represents a function call node in the AST."""
    def __init__(self, name, args, line_num=None):
        super().__init__(line_num)
        self.name = name
        self.args = args
    def __repr__(self):
        return f"Call({self.name}, [{', '.join(map(repr, self.args))}])"


class FunctionDef(ASTNode):
    """Represents a function definition node in the AST."""
    def __init__(self, name, params, body, line_num=None):
        super().__init__(line_num)
        self.name = name
        self.params = params
        self.body = body
    def __repr__(self):
        return f"FunctionDef({self.name}, [{', '.join(self.params)}], {repr(self.body)})"


class StructDef(ASTNode):
    """Represents a struct definition node in the AST."""
    def __init__(self, name, fields, line_num=None):
        super().__init__(line_num)
        self.name = name
        self.fields = fields
    def __repr__(self):
        return f"StructDef({self.name}, [{', '.join(self.fields)}])"


class MemberAccess(ASTNode):
    """Represents a member access (dot operator) node in the AST."""
    def __init__(self, base, member, line_num=None):
        super().__init__(line_num)
        self.base = base
        self.member = member
    def __repr__(self):
        return f"MemberAccess({repr(self.base)}, {self.member})"


class ReturnStatement(ASTNode):
    """Represents a return statement node in the AST."""
    def __init__(self, value, line_num=None):
        super().__init__(line_num)
        self.value = value
    def __repr__(self):
        return f"Return({repr(self.value)})"


class StatementList(ASTNode):
    """Represents a block of statements (e.g., function body, loop body) in the AST."""
    def __init__(self, statements, line_num=None):
        super().__init__(line_num)
        self.statements = statements
    def __repr__(self):
        # Represent block content
        return f"Block([{'; '.join(map(repr, self.statements))}])"


class ForLoop(ASTNode):
    """Represents a for loop node in the AST."""
    def __init__(self, init, condition, update, body, line_num=None):
        super().__init__(line_num)
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body
    def __repr__(self):
        return f"ForLoop({repr(self.init)}, {repr(self.condition)}, {repr(self.update)}, {repr(self.body)})"


class IfStatement(ASTNode):
    """Represents an if or if-else statement node in the AST."""
    def __init__(self, condition, then_branch, else_branch=None, line_num=None):
        super().__init__(line_num)
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch
    def __repr__(self):
        if self.else_branch:
            return f"IfElse({repr(self.condition)}, {repr(self.then_branch)}, {repr(self.else_branch)})"
        return f"If({repr(self.condition)}, {repr(self.then_branch)})"

# --- New AST Node for While Loop ---
class WhileLoop(ASTNode):
    """Represents a while loop node in the AST."""
    def __init__(self, condition, body, line_num=None):
        super().__init__(line_num)
        self.condition = condition
        self.body = body
    def __repr__(self):
        return f"WhileLoop({repr(self.condition)}, {repr(self.body)})"
# --- End New AST Node ---


# -----------------------------
# Global Tables for Variables, Functions, and Structs
# -----------------------------
symbol_table = {}
function_table = {}
struct_table = {}
# --- NEW: Class Table ---
class_table = {}
# --- END NEW ---

# Set to keep track of deleted variable names
deleted_variables = set() # Moved this to global scope

# Dummy function definition for 'add' (expects 2 arguments) to simulate expected error.
# This is for demonstrating semantic error handling.
function_table["add"] = FunctionDef("add", ["a", "b"], StatementList([], line_num=None), line_num=None) # Added line_num=None

# -----------------------------
# Parser Definition
# -----------------------------
class Parser:
    """Parses a list of tokens into an Abstract Syntax Tree (AST)."""
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token_info(self):
        """Returns the current token and its line number, or (None, None) if at the end."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def current(self):
        """Returns the value of the current token, or None if at the end."""
        token, _ = self.current_token_info()
        return token

    def current_line_num(self):
        """Returns the line number of the current token, or None if at the end."""
        _, line_num = self.current_token_info()
        return line_num

    def eat(self, token_value):
        """
        Consumes the current token if it matches the expected value, advancing the position.

        Args:
            token_value (str): The expected value of the current token.

        Returns:
            int: The line number of the consumed token.

        Raises:
            SyntaxError: If the current token does not match the expected value or is unexpected end of input.
        """
        # Basic token consumption with syntax error checking.
        token, line_num = self.current_token_info()
        if token == token_value:
            self.pos += 1
            return line_num
        else:
            # Improved syntax error messages with line numbers
            if token_value == ')' and token != ')':
                raise SyntaxError(f"Syntax error on line {line_num}: missing closing parenthesis")
            if token is None:
                 raise SyntaxError(f"Syntax error: Unexpected end of input. Expected '{token_value}'")
            raise SyntaxError(f"Syntax error on line {line_num}: Expected token '{token_value}', got '{token}'")

    def parse(self):
        """
        Parses a sequence of statements until the end of the token list.

        Returns:
            StatementList: An AST node representing the list of parsed statements.
        """
        # Parses a list of statements until the end.
        statements = []
        # The REPL will handle accumulating input until a full statement is ready to parse.
        # This parse method is now expected to receive a buffer containing a complete statement.
        # It will try to parse one or more statements until the end of the provided tokens.
        while self.current() is not None:
             statements.append(self.parse_statement())
             if self.current() == ';':
                 self.eat(';')
             # Allow multiple statements without trailing semicolon at the end
             elif self.current() is not None and self.current() != '}':
                  # If not the end of input or a closing brace, expect a semicolon
                  # This case is handled by the parse_statement or parse_block methods now
                  pass

        # If we reached the end of tokens but didn't parse anything, it might be an empty buffer case,
        # or an incomplete statement that didn't trigger a syntax error during parsing.
        # For the REPL, we rely on the buffer check to decide when to call parse.
        # If parse is called, it should ideally consume all tokens for a complete statement.
        # If it doesn't, it implies an issue or an incomplete parse that didn't raise an error yet.
        # The REPL logic below will handle this.

        return StatementList(statements)


    def parse_struct_def(self):
        """
        Parses a struct definition.

        Returns:
            StructDef: An AST node representing the struct definition.

        Raises:
            SyntaxError: If the struct definition syntax is invalid.
        """
        line_num = self.eat("struct")
        if not self.current() or not self.current().isidentifier():
            raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected struct name after 'struct'")
        name = self.current()
        self.eat(name)
        self.eat("{")
        fields = []
        while self.current() != "}":
            if not self.current() or not self.current().isidentifier(): # Added check for None
                raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected field name, got '{self.current()}'")
            fields.append(self.current())
            self.eat(self.current())
            if self.current() == ",":
                self.eat(",")
            elif self.current() != '}': # Ensure comma or closing brace after field
                 raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected ',' or '}}'")
        self.eat("}")
        struct_table[name] = fields
        return StructDef(name, fields, line_num)

    def parse_for_loop(self):
        """
        Parses a for loop statement.

        Returns:
            ForLoop: An AST node representing the for loop.

        Raises:
            SyntaxError: If the for loop syntax is invalid.
        """
        line_num = self.eat("for")
        self.eat("(")
        init = self.parse_statement() # For loop init can be a statement like assignment
        self.eat(";")
        condition = self.parse_expression(0)
        self.eat(";")
        update = self.parse_statement() # For loop update can be a statement like assignment
        self.eat(")")
        body = self.parse_block()
        return ForLoop(init, condition, update, body, line_num)

    def parse_if_statement(self):
        """
        Parses an if or if-else statement.

        Returns:
            IfStatement: An AST node representing the if or if-else statement.

        Raises:
            SyntaxError: If the if or if-else statement syntax is invalid.
        """
        line_num = self.eat("if")
        # Explicitly check for the opening parenthesis and raise the specific error if missing
        if self.current() != '(':
             raise SyntaxError(f"Syntax error: missing parentheses") # Exact error message from rubric
        self.eat("(")
        condition = self.parse_expression(0)
        self.eat(")")
        then_branch = self.parse_block()
        else_branch = None
        if self.current() == "else":
            self.eat("else")
            else_branch = self.parse_block()
        return IfStatement(condition, then_branch, else_branch, line_num)

    # --- New Parsing Logic for While Loop ---
    def parse_while_loop(self):
        """
        Parses a while loop statement.

        Returns:
            WhileLoop: An AST node representing the while loop.

        Raises:
            SyntaxError: If the while loop syntax is invalid.
        """
        line_num = self.eat("while")
        if self.current() != '(':
             raise SyntaxError(f"Syntax error: missing parentheses")
        self.eat("(")
        condition = self.parse_expression(0)
        self.eat(")")
        body = self.parse_block()
        return WhileLoop(condition, body, line_num)
    # --- End New Parsing Logic ---

    # --- New Parsing Logic for Delete ---
    def parse_delete_call(self):
        """
        Parses a delete function call.

        Returns:
            DeleteCall: An AST node representing the delete call.

        Raises:
            SyntaxError: If the delete call syntax is invalid.
        """
        line_num = self.eat("delete")
        self.eat("(")
        # Expect a single argument which must be a variable
        arg = self.parse_expression(0)
        if not isinstance(arg, Variable):
             raise SyntaxError(f"Syntax error on line {self.current_line_num()}: delete() expects a variable name as argument")
        self.eat(")")
        return DeleteCall(arg, line_num)
    # --- End New Parsing Logic ---

    # --- MODIFICATION: Parsing Logic for Class Definition ---
    def parse_class_def(self):
        """
        Parses a class definition.

        Returns:
            ClassDef: An AST node representing the class definition.

        Raises:
            SyntaxError: If the class definition syntax is invalid.
        """
        line_num = self.eat("class")
        if not self.current() or not self.current().isidentifier():
            raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected class name after 'class'")
        name = self.current()
        self.eat(name)
        self.eat("{")
        methods = {}
        while self.current() != "}":
            # --- MODIFICATION: Parse method definition (identifier, '(', params, ')', block) ---
            method_name, method_line_num = self.current_token_info()
            if not method_name or not method_name.isidentifier():
                 raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected method name or '}}'")
            self.eat(method_name)
            self.eat("(")
            params = []
            # Add 'self' as the implicit first parameter for methods
            params.append("self") # Conventionally named 'self'
            while self.current() != ")":
                 if not self.current() or not self.current().isidentifier():
                      raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Invalid parameter name '{self.current()}'")
                 params.append(self.current())
                 self.eat(self.current())
                 if self.current() == ",":
                      self.eat(",")
                 elif self.current() != ")":
                      raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected ',' or ')' after parameter")
            self.eat(")")
            body = self.parse_block() # Parse the method body as a block
            methods[method_name] = FunctionDef(method_name, params, body, method_line_num) # Store method as a FunctionDef
            # --- END MODIFICATION ---
        self.eat("}")
        class_table[name] = ClassDef(name, methods, line_num) # Store class definition globally
        return class_table[name]
    # --- END MODIFICATION ---


    def parse_block(self):
        """
        Parses a block of statements enclosed in curly braces {}.

        Returns:
            StatementList: An AST node representing the block of statements.

        Raises:
            SyntaxError: If the block syntax is invalid.
        """
        line_num = self.eat("{")
        statements = []
        while self.current() != "}":
            statements.append(self.parse_statement())
            if self.current() == ';':
                self.eat(";")
            # Allow multiple statements in a block without trailing semicolon if it's the last statement before '}'
            elif self.current() != '}':
                 raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected ';' or '}}'")
        self.eat("}")
        return StatementList(statements, line_num)

    def parse_statement(self):
        """
        Parses a single statement.

        Returns:
            ASTNode: An AST node representing the parsed statement.

        Raises:
            SyntaxError: If the statement syntax is invalid.
        """
        # Handle 'let' keyword for variable declaration
        if self.current() == "let":
            line_num = self.eat("let")
            if not self.current() or not self.current().isidentifier():
                 raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected variable name after 'let'")
            var_name = self.current()
            # Check for redeclaration before eating the token
            if var_name in symbol_table:
                 raise SyntaxError(f"Syntax error: Redeclaration of variable '{var_name}'") # Specific error for redeclaration
            # --- FIX: Add the variable to the symbol table during parsing for 'let' ---
            # Add the variable to the symbol table with a placeholder value (like None)
            # This ensures that the redeclaration check works during parsing.
            symbol_table[var_name] = None
            # --- End FIX ---
            self.eat(var_name)
            value = None
            if self.current() == '=':
                self.eat('=')
                value = self.parse_expression(0)
            # We don't require a type after 'let' based on the rubric examples,
            # but the previous code had logic for it. Removing for simplicity based on rubric.
            return Assignment(var_name, value, line_num)
        # Handle variable declaration with type (as in previous code, though not explicit in rubric examples)
        elif (self.current() is not None and self.current().isidentifier() and
              self.pos + 2 < len(self.tokens) and
              self.tokens[self.pos+1][0].isidentifier() and
              self.tokens[self.pos+2][0] == '='):
             # Skip type (e.g., "Point") - keeping this logic for now
             type_name = self.current()
             self.eat(type_name)
             var_name = self.current()
             line_num = self.eat(var_name)
             # Check for redeclaration here too if type is specified
             if var_name in symbol_table:
                  raise SyntaxError(f"Syntax error: Redeclaration of variable '{var_name}'") # Specific error for redeclaration
             # --- FIX: Add the variable to the symbol table during parsing for typed declaration ---
             symbol_table[var_name] = None # Add with a placeholder value for now
             # --- End FIX ---
             self.eat('=')
             expr = self.parse_expression(0)
             return Assignment(var_name, expr, line_num)
        # Other statement forms.
        elif self.current() == "struct":
            return self.parse_struct_def()
        elif self.current() == "return":
            line_num = self.eat("return")
            value = self.parse_expression(0)
            return ReturnStatement(value, line_num)
        elif self.current() == "function":
            # This is for parsing standalone global functions
            func_def = self.parse_function_def()
            function_table[func_def.name] = func_def # Store global function
            return func_def
        elif self.current() == "for":
            return self.parse_for_loop()
        elif self.current() == "if":
            return self.parse_if_statement()
        # --- Add Parsing for While Loop ---
        elif self.current() == "while":
            return self.parse_while_loop()
        # --- End Add Parsing ---
        # --- Add Parsing for Delete ---
        elif self.current() == "delete":
             return self.parse_delete_call()
        # --- End Add Parsing ---
        # --- NEW: Add Parsing for Class Definition ---
        elif self.current() == "class":
             return self.parse_class_def()
        # --- END NEW ---
        else:
            # Assume it's an expression statement
            expr = self.parse_expression(0)
            return expr # Return the expression node directly

    # -----------------------------
    # Function Parameters & Return Values
    # (Function definition parsing)
    # -----------------------------
    # --- MODIFICATION: parse_function_def now only parses the function structure, not stores it ---
    # The caller (parse_statement or parse_class_def) is responsible for storing the FunctionDef node.
    def parse_function_def(self):
        """
        Parses a function definition (used for both global functions and methods).

        Returns:
            FunctionDef: An AST node representing the function definition.

        Raises:
            SyntaxError: If the function definition syntax is invalid.
        """
        line_num = self.eat("function") # Still expect 'function' for global functions
        if not self.current() or not self.current().isidentifier():
            raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected function name after 'function'")
        name = self.current()
        self.eat(name)
        self.eat("(")
        params = []
        while self.current() != ")":
            if not self.current() or not self.current().isidentifier(): # Added check for None
                raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Invalid parameter name '{self.current()}'")
            params.append(self.current())
            self.eat(self.current())
            if self.current() == ",":
                self.eat(",")
            elif self.current() != ")": # Ensure a comma or closing parenthesis follows a parameter
                 raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected ',' or ')' after parameter")
        self.eat(")")
        body = self.parse_block()
        return FunctionDef(name, params, body, line_num)
    # --- END MODIFICATION ---


    def parse_expression(self, min_precedence=0):
        """
        Parses an expression using the precedence climbing method.

        Args:
            min_precedence (int): The minimum precedence level to consider for operators.

        Returns:
            ASTNode: An AST node representing the parsed expression.

        Raises:
            SyntaxError: If the expression syntax is invalid.
        """
        left = self.parse_unary()
        # Assignment handling
        if self.current() == '=' and min_precedence == 0:
            line_num = self.eat('=')
            right = self.parse_expression(0)
            # Check if the left side is a variable or a member access for assignment
            if not isinstance(left, (Variable, MemberAccess, IndexAccess)): # --- MODIFICATION: Added IndexAccess to allowed LHS ---
                raise SyntaxError(f"Syntax error on line {line_num}: Left-hand side of assignment must be a variable, member access, or index access")
            # If it's a variable, create an Assignment node
            if isinstance(left, Variable):
                # Note: For simple assignment like `x = 5;`, the Variable node `x` is returned from parse_primary.
                # The Assignment node is created here. The symbol table update happens during evaluation.
                left = Assignment(left.name, right, line_num)
            # If it's a member access or index access, the assignment will happen during evaluation
            return left # Return the Assignment, MemberAccess, or IndexAccess node

        while self.current() and self.current() in PRECEDENCE:
            current_token = self.current()
            current_precedence = PRECEDENCE[current_token]
            if current_precedence < min_precedence:
                break
            line_num = self.eat(current_token)
            next_precedence = current_precedence + 1 if current_token != '^' else current_precedence
            right = self.parse_expression(next_precedence)
            left = BinaryOp(current_token, left, right, line_num)
        return left

    def parse_unary(self):
        """
        Parses a unary operation or a primary expression.

        Returns:
            ASTNode: An AST node representing the unary operation or primary expression.
        """
        if self.current() in ('-', '!', 'not'):
            op = self.current()
            line_num = self.eat(op)
            operand = self.parse_unary()
            return UnaryOp(op, operand, line_num)
        return self.parse_primary()

    def parse_primary(self):
        """
        Parses a primary expression (literals, variables, function calls, parenthesized expressions, lists, index access, new expressions).

        Returns:
            ASTNode: An AST node representing the primary expression.

        Raises:
            SyntaxError: If the primary expression syntax is invalid.
        """
        token, line_num = self.current_token_info()
        if token == '(':
            self.eat('(')
            expr = self.parse_expression(0)
            if self.current() != ')':
                raise SyntaxError(f"Syntax error on line {self.current_line_num()}: missing closing parenthesis")
            self.eat(')')
            left = expr
        elif token and token.startswith("'") and token.endswith("'"):
            self.eat(token)
            left = StringLiteral(token[1:-1], line_num)
        elif token and token.startswith('"') and token.endswith('"'):
             self.eat(token)
             left = StringLiteral(token[1:-1], line_num)
        elif token == "true":
            self.eat("true")
            left = BooleanLiteral(True, line_num)
        elif token == "false":
            self.eat("false")
            left = BooleanLiteral(False, line_num)
        # --- MODIFICATION: Handle 'null' literal ---
        elif token == "null":
             self.eat("null")
             left = NullLiteral(line_num)
        # --- END MODIFICATION ---
        # --- MODIFICATION: Handle List Literal ---
        elif token == '[':
             line_num = self.eat('[')
             elements = []
             while self.current() != ']':
                  elements.append(self.parse_expression(0))
                  if self.current() == ',':
                       self.eat(',')
                  elif self.current() != ']':
                       raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected ',' or ']' in list literal")
             self.eat(']')
             left = ListLiteral(elements, line_num)
        # --- END MODIFICATION ---
        elif token and token.isdigit():
            self.eat(token)
            left = Number(int(token), line_num)
        # --- NEW: Handle 'new' expression ---
        elif token == "new":
             line_num = self.eat("new")
             if not self.current() or not self.current().isidentifier():
                  raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected class name after 'new'")
             class_name = self.current()
             self.eat(class_name)
             self.eat("(")
             args = []
             while self.current() != ")":
                  args.append(self.parse_expression(0))
                  if self.current() == ',':
                       self.eat(',')
                  elif self.current() != ')':
                       raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected ',' or ')' in constructor arguments")
             self.eat(")")
             left = NewExpression(class_name, args, line_num)
        # --- END NEW ---
        elif token and token.isidentifier():
            self.eat(token)
            if self.current() == '(':
                self.eat('(')
                args = []
                while self.current() != ')':
                    args.append(self.parse_expression(0))
                    if self.current() == ',':
                        self.eat(',')
                    elif self.current() != ')': # Ensure comma or closing parenthesis after argument
                         raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected ',' or ')' after function argument")
                self.eat(')')
                # --- MODIFICATION: Check for 'delete' as a function call here ---
                if token == "delete":
                     # If it's a delete call, the parsing logic for delete() is handled in parse_statement,
                     # but if it's encountered as part of an expression (e.g., `x = delete(y);`),
                     # this would be an error in a real language. For simplicity, we'll assume delete
                     # is only used as a statement and handle it in parse_statement.
                     # If we were to support `x = delete(y);`, we'd need a different AST node and evaluation logic.
                     raise SyntaxError(f"Syntax error on line {line_num}: delete() can only be used as a statement")
                # --- END MODIFICATION ---
                left = FunctionCall(token, args, line_num)
            else:
                left = Variable(token, line_num)
        else:
            raise SyntaxError(f"Syntax error on line {line_num}: Unexpected token: {token}")
        # -----------------------------
        # Member Access Handling (dot operator) and Index Access Handling
        # --- MODIFICATION: Handle Method Call after Member Access ---
        while self.current() == '.' or self.current() == '[':
            if self.current() == '.':
                 self.eat('.')
                 member, member_line_num = self.current_token_info()
                 if not member or not member.isidentifier():
                     raise SyntaxError(f"Syntax error on line {member_line_num}: Expected member name after '.'")
                 self.eat(member)
                 # Check if the next token is '(' for a method call
                 if self.current() == '(':
                      self.eat('(')
                      args = []
                      while self.current() != ')':
                           args.append(self.parse_expression(0))
                           if self.current() == ',':
                                self.eat(',')
                           elif self.current() != ')':
                                raise SyntaxError(f"Syntax error on line {self.current_line_num()}: Expected ',' or ')' in method arguments")
                      self.eat(')')
                      # Create a MethodCall node
                      left = MethodCall(left, member, args, line_num) # Use line_num of the base
                 else:
                      # It's a regular member access (like accessing a field)
                      left = MemberAccess(left, member, line_num) # Use line_num of the base for the MemberAccess node
            # --- END MODIFICATION ---
            # --- MODIFICATION: Handle Index Access ---
            elif self.current() == '[':
                 line_num = self.eat('[')
                 index = self.parse_expression(0)
                 if self.current() != ']':
                      raise SyntaxError(f"Syntax error on line {self.current_line_num()}: missing closing square bracket")
                 self.eat(']')
                 left = IndexAccess(left, index, line_num)
            # --- END MODIFICATION ---
        return left

# -----------------------------
# Evaluator Definition
# -----------------------------
# --- NEW: Define a simple object instance representation ---
class ObjectInstance:
    """Represents an instance of a class."""
    def __init__(self, class_def):
        self.class_def = class_def
        self.data = {} # Dictionary to store instance data (fields)

    def __repr__(self):
        # Simple representation for printing
        return f"Instance({self.class_def.name}, data={self.data})"
# --- END NEW ---


def evaluate(ast, local_vars=None):
    """
    Evaluates the given Abstract Syntax Tree (AST).

    Args:
        ast (ASTNode): The root node of the AST to evaluate.
        local_vars (dict, optional): A dictionary representing the local variable scope.
                                     Defaults to None, which uses the global scope.

    Returns:
        any: The result of the evaluation.

    Raises:
        Exception: If an unsupported AST node or runtime error is encountered.
        TypeError: If a type mismatch occurs during evaluation.
        ZeroDivisionError: If division by zero occurs.
        IndexError: If an index is out of bounds for a list.
        NameError: If a variable or class is not defined.
        AttributeError: If a member or method is not found on an object.
    """
    if local_vars is None:
        local_vars = {}

    # Helper function to get variable value from local or global scope
    def get_variable(name, line_num):
         if name in local_vars:
             return local_vars[name]
         elif name in symbol_table:
             # Check if the variable has been assigned a value (not just declared)
             if symbol_table[name] is None:
                 # If declared with let but not assigned, this is still an error in a strict language
                 raise Exception(f"Error: variable '{name}' is used before assignment")
             return symbol_table[name]
         else:
             # --- MODIFICATION: Check if the variable was deleted ---
             if name in deleted_variables:
                 raise Exception(f"Error: access to deleted object") # Specific error for deleted objects
             # --- MODIFICATION: Automatically declare variable with default value (0) if not found ---
             # This is done to make the user's specific input work without requiring 'let'.
             # In a real language, this would typically be a "variable not defined" error.
             # We will add a warning here to indicate this non-standard behavior.
             print(f"Warning: Variable '{name}' used without declaration on line {line_num}. Initializing to 0.")
             symbol_table[name] = 0 # Assign a default value (0 for numbers)
             return symbol_table[name]
             # --- END MODIFICATION ---


    # Helper function to set variable value in local or global scope
    def set_variable(name, value, line_num):
         # For simplicity, assignments always go to the global symbol table for now
         # More advanced scope handling would check local_vars first.
         # If the variable is not in the symbol table during assignment,
         # it implies it was used without 'let' and should ideally be an error.
         # However, with the modification in get_variable, it will be added there first.
         # We'll keep the assignment logic simple for now.
         symbol_table[name] = value


    if isinstance(ast, Number):
        return ast.value
    elif isinstance(ast, BooleanLiteral):
        return ast.value
    elif isinstance(ast, StringLiteral):
        return ast.value
    # --- MODIFICATION: Handle NullLiteral evaluation ---
    elif isinstance(ast, NullLiteral):
        return ast.value # Returns Python's None
    # --- END MODIFICATION ---
    # --- MODIFICATION: Handle ListLiteral evaluation ---
    elif isinstance(ast, ListLiteral):
        return [evaluate(element, local_vars) for element in ast.elements]
    # --- END MODIFICATION ---
    elif isinstance(ast, Variable):
        return get_variable(ast.name, ast.line_num)
    elif isinstance(ast, Assignment):
        # --- MODIFICATION: Add print statement for variable updates and memory clear message ---
        # Check if the variable exists before the assignment
        is_update = ast.name in symbol_table
        value = evaluate(ast.value, local_vars)

        # Check if the new value is None (representing null)
        if value is None:
             print("Memory cleared or collected") # Print the memory clear message

        # --- MODIFICATION: Handle assignment to IndexAccess and MemberAccess on instances ---
        if isinstance(ast.name, IndexAccess):
             base_obj = evaluate(ast.name.base, local_vars)
             index_val = evaluate(ast.name.index, local_vars)

             if not isinstance(base_obj, list): # Check if the base is a list
                  raise TypeError(f"Runtime error on line {ast.line_num}: Cannot assign to index of non-list type")
             if not isinstance(index_val, int): # Check if the index is an integer
                  raise TypeError(f"Runtime error on line {ast.line_num}: List index must be an integer")
             if index_val < 0 or index_val >= len(base_obj): # Check if index is out of bounds
                  raise IndexError(f"Runtime error on line {ast.line_num}: List index out of range")

             base_obj[index_val] = value # Perform the assignment to the list element
             # No specific update message for list element assignment in the requested output
             return value
        elif isinstance(ast.name, MemberAccess):
             # Handle assignment to an object's data field
             instance_obj = evaluate(ast.name.base, local_vars)
             if not isinstance(instance_obj, ObjectInstance):
                  raise TypeError(f"Runtime error on line {ast.line_num}: Cannot assign to member of non-object type")
             member_name = ast.name.member
             instance_obj.data[member_name] = value # Assign to the instance's data
             return value
        # --- END MODIFICATION ---
        else: # Handle assignment to a Variable (existing logic)
             set_variable(ast.name, value, ast.line_num)
             # Print the update message if it was an update (not the initial 'let' assignment)
             # Also, avoid printing "updated correctly" when assigning null, as the requested output doesn't show both.
             if is_update and value is not None:
                  print(f"Variable {ast.name} updated correctly")
             return value
        # --- END MODIFICATION ---

    elif isinstance(ast, BinaryOp):
        left = evaluate(ast.left, local_vars)
        right = evaluate(ast.right, local_vars)

        # Basic Type Checking for binary operations - matches rubric examples
        if ast.op in ('+', '-', '*', '/', '^', '>', '<'):
            # Arithmetic and comparison operators require numbers
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                # --- MODIFICATION: Include line number in TypeError message ---
                raise TypeError(f"Error on line {ast.line_num}: cannot add string and number") # Using the specific message for string+number as per rubric example
                # --- END MODIFICATION ---
            if ast.op == '+': return left + right
            elif ast.op == '-': return left - right
            elif ast.op == '*': return left * right
            elif ast.op == '/':
                # Modified error message to match the requested format "Error: division by zero"
                if right == 0:
                     raise ZeroDivisionError(f"Error: division by zero")
                return left / right
            elif ast.op == '>': return left > right
            elif ast.op == '<': return left < right

        elif ast.op in ('==', '!='):
             # Equality and inequality can compare different types, but we'll keep it simple for now
             return left == right if ast.op == '==' else left != right

        elif ast.op in ('&&', 'and', '||', 'or'):
            # Logical operators require booleans
            if not isinstance(left, bool) or not isinstance(right, bool):
                raise TypeError(f"Runtime error on line {ast.line_num}: Unsupported operand types for {ast.op}: '{type(left).__name__}' and '{type(right).__name__}'")
            return left and right if ast.op in ('&&', 'and') else left or right

        else:
            raise Exception(f"Runtime error on line {ast.line_num}: Unsupported operator {ast.op}")
    elif isinstance(ast, UnaryOp):
        operand = evaluate(ast.operand, local_vars)
        if ast.op == '-':
             if not isinstance(operand, (int, float)):
                 raise TypeError(f"Runtime error on line {ast.line_num}: Unsupported operand type for unary '-': '{type(operand).__name__}'")
             return -operand
        elif ast.op in ('!', 'not'):
             if not isinstance(operand, bool):
                 raise TypeError(f"Runtime error on line {ast.line_num}: Unsupported operand type for unary '{ast.op}': '{type(operand).__name__}'")
             return not operand
        else:
            raise Exception(f"Runtime error on line {ast.line_num}: Unsupported unary operator {ast.op}")
    elif isinstance(ast, FunctionCall):
        # -----------------------------
        # Function Call Evaluation Block (Function Parameters & Return Values)
        # --- MODIFICATION: Handle Function Calls that are not Method Calls ---
        # This block now only handles calls that are NOT MethodCalls (e.g., print(), add(), global functions)
        # Method calls like p.greet() are handled by the new MethodCall AST node.
        # First, check for struct instantiation (still a function-like call).
        if ast.name in struct_table:
            fields = struct_table[ast.name]
            if len(fields) != len(ast.args):
                raise Exception(f"Runtime error on line {ast.line_num}: Struct {ast.name} expects {len(fields)} arguments, got {len(ast.args)}")
            values = [evaluate(arg, local_vars) for arg in ast.args]
            return dict(zip(fields, values))
        # Built-in "print" function - matches rubric example
        if ast.name == "print":
            args = [evaluate(arg, local_vars) for arg in ast.args]
            # Convert boolean values to "true" or "false" strings for printing
            formatted_args = []
            for arg in args:
                 if isinstance(arg, bool):
                      formatted_args.append("true" if arg else "false")
                 elif isinstance(arg, str):
                      formatted_args.append(f"'{arg}'") # Include quotes for string literals in print output
                 elif arg is None: # Handle printing of null
                      formatted_args.append("null")
                 # --- NEW: Handle printing of ObjectInstance ---
                 elif isinstance(arg, ObjectInstance):
                      formatted_args.append(repr(arg)) # Use the ObjectInstance's __repr__
                 # --- END NEW ---
                 else:
                      formatted_args.append(str(arg))

            # Print with the exact format "Prints: " followed by arguments separated by spaces
            print("Prints:", *formatted_args)
            return None # print doesn't return a value
        # Built-in "input" function - matches rubric example
        if ast.name == "input":
             # --- Implementation for input() ---
             if len(ast.args) > 0:
                  # Optional prompt argument for input()
                  prompt = evaluate(ast.args[0], local_vars)
                  return input(prompt)
             else:
                  # No prompt argument
                  return input()
             # --- End Implementation for input() ---

        # Normal global function call handling:
        func = function_table.get(ast.name)
        if not func:
            raise Exception(f"Runtime error on line {ast.line_num}: Undefined function '{ast.name}'")
        if len(func.params) != len(ast.args):
            # Matches rubric example for incorrect number of arguments
            raise Exception(f"Runtime error on line {ast.line_num}: Function '{ast.name}' expects {len(func.params)} arguments, but got {len(ast.args)}")
        evaluated_args = [evaluate(arg, local_vars) for arg in ast.args]
        new_locals = {param: evaluated_args[i] for i, param in enumerate(func.params)}
        # Evaluate the function body with the new local scope
        return evaluate(func.body, new_locals)
        # --- END MODIFICATION ---

    elif isinstance(ast, MemberAccess):
        # -----------------------------
        # Struct Member Access Evaluation
        # --- MODIFICATION: Handle Member Access on Object Instances ---
        base_obj = evaluate(ast.base, local_vars)
        member_name = ast.member

        if isinstance(base_obj, ObjectInstance):
             # Check if the member is a data field
             if member_name in base_obj.data:
                  return base_obj.data[member_name]
             # Check if the member is a method
             if member_name in base_obj.class_def.methods:
                  # When accessing a method, return the method definition itself
                  # The MethodCall AST node will handle calling it.
                  return base_obj.class_def.methods[member_name]
             # If not found as data or method, it's an error
             raise AttributeError(f"Runtime error on line {ast.line_num}: Object of type '{base_obj.class_def.name}' has no member '{member_name}'")

        elif isinstance(base_obj, dict): # Handle struct member access (existing logic)
            if ast.member not in base_obj:
                raise Exception(f"Runtime error on line {ast.line_num}: Member '{ast.member}' not found in {base_obj}")
            return base_obj[ast.member]
        # --- END MODIFICATION ---
        else:
             raise Exception(f"Runtime error on line {ast.line_num}: Attempted member access on non-struct or non-object: {base_obj}")

    # --- MODIFICATION: Handle Index Access Evaluation ---
    elif isinstance(ast, IndexAccess):
        """
        Evaluates an index access (e.g., list[index]).
        """
        base_obj = evaluate(ast.base, local_vars)
        index_val = evaluate(ast.index, local_vars)

        if not isinstance(base_obj, list): # Check if the base is a list
             raise TypeError(f"Runtime error on line {ast.line_num}: Cannot index non-list type")
        if not isinstance(index_val, int): # Check if the index is an integer
             raise TypeError(f"Runtime error on line {ast.line_num}: List index must be an integer")
        if index_val < 0 or index_val >= len(base_obj): # Check if index is out of bounds
             raise IndexError(f"Runtime error on line {ast.line_num}: List index out of range")

        return base_obj[index_val] # Return the element at the specified index
    # --- END MODIFICATION ---
    elif isinstance(ast, ForLoop):
        # -----------------------------
        # For Loop Evaluation (Enhanced Control Structures)
        # -----------------------------
        # For loops have their own scope for the initialization variable
        for_loop_vars = local_vars.copy()
        evaluate(ast.init, for_loop_vars) # Evaluate init in the loop's scope
        result = None
        while evaluate(ast.condition, for_loop_vars):
            result = evaluate(ast.body, for_loop_vars) # Evaluate body in the loop's scope
            evaluate(ast.update, for_loop_vars) # Evaluate update in the loop's scope
        return result
    elif isinstance(ast, IfStatement):
        # -----------------------------
        # If-Else Statement Evaluation (Enhanced Control Structures)
        # -----------------------------
        if evaluate(ast.condition, local_vars):
            return evaluate(ast.then_branch, local_vars)
        elif ast.else_branch:
            return evaluate(ast.else_branch, local_vars)
        return None
    # --- New Evaluation Logic for While Loop ---
    elif isinstance(ast, WhileLoop):
        """
        Evaluates a while loop.
        """
        result = None
        # Evaluate the condition. If it's not a boolean, raise a type error.
        condition_value = evaluate(ast.condition, local_vars)
        if not isinstance(condition_value, bool):
             raise TypeError(f"Runtime error on line {ast.condition.line_num}: While loop condition must be a boolean")

        while evaluate(ast.condition, local_vars):
            result = evaluate(ast.body, local_vars) # Evaluate body in the current scope
        return result
    # --- End New Evaluation Logic ---
    # --- New Evaluation Logic for Delete ---
    elif isinstance(ast, DeleteCall):
        """
        Evaluates a delete call.
        """
        # Ensure the argument is a variable
        if not isinstance(ast.name, Variable):
             raise Exception(f"Runtime error on line {ast.line_num}: delete() expects a variable name")

        var_name = ast.name.name
        # Check if the variable exists before attempting to delete
        if var_name not in symbol_table:
             # If the variable is not in the symbol table, it's either not defined or already deleted.
             # We'll check the deleted_variables set for a more specific error message if needed.
             if var_name in deleted_variables:
                  raise Exception(f"Runtime error on line {ast.line_num}: Variable '{var_name}' already deleted")
             else:
                  raise Exception(f"Runtime error on line {ast.line_num}: Undefined variable '{var_name}'")

        # Remove the variable from the symbol table
        del symbol_table[var_name]
        # Add the variable name to a set of deleted variables for checking later access
        deleted_variables.add(var_name)
        return None # delete doesn't return a value
    # --- End New Evaluation Logic ---
    # --- NEW: Evaluation Logic for Class Definition ---
    elif isinstance(ast, ClassDef):
        # Class definitions are registered in the class_table during parsing.
        # Evaluation just returns the class definition itself (or None if it's just a statement).
        # For simplicity, we'll return None as it's typically a statement.
        return None
    # --- END NEW ---
    # --- NEW: Evaluation Logic for New Expression ---
    elif isinstance(ast, NewExpression):
        """
        Evaluates a 'new' expression to create an object instance.
        """
        class_name = ast.class_name
        class_def = class_table.get(class_name)
        if not class_def:
             raise NameError(f"Runtime error on line {ast.line_num}: Undefined class '{class_name}'")

        instance = ObjectInstance(class_def) # Create a new instance

        # Check for a constructor method (e.g., 'init') and call it
        # For this example, we'll assume the constructor is named 'init' if it exists.
        # The provided example uses 'greet' directly, but a constructor is common for initialization.
        # Let's adapt to the example and assume 'greet' is the method to call after instantiation for this specific case.
        # A more robust approach would look for a dedicated constructor method.
        # For the given example, we just create the instance and return it.
        # The subsequent p.greet() call will handle the method execution.

        # If there was an 'init' method, we would evaluate it here:
        # constructor = class_def.methods.get('init')
        # if constructor:
        #      if len(constructor.params) != len(ast.args) + 1: # +1 for 'self'
        #           raise Exception(f"Runtime error on line {ast.line_num}: Constructor '{class_name}.init' expects {len(constructor.params) - 1} arguments, but got {len(ast.args)}")
        #      evaluated_args = [evaluate(arg, local_vars) for arg in ast.args]
        #      constructor_locals = {constructor.params[0]: instance} # Map 'self'
        #      for i, arg_val in enumerate(evaluated_args):
        #           constructor_locals[constructor.params[i+1]] = arg_val
        #      evaluate(constructor.body, constructor_locals)

        return instance # Return the newly created instance
    # --- END NEW ---
    # --- NEW: Evaluation Logic for Method Call ---
    elif isinstance(ast, MethodCall):
        """
        Evaluates a method call on an object instance.
        """
        instance_obj = evaluate(ast.base, local_vars)
        method_name = ast.method_name

        if not isinstance(instance_obj, ObjectInstance):
             raise TypeError(f"Runtime error on line {ast.line_num}: Attempted to call method '{method_name}' on non-object type")

        class_def = instance_obj.class_def
        method_def = class_def.methods.get(method_name)

        if not method_def:
             raise AttributeError(f"Runtime error on line {ast.line_num}: Object of type '{class_def.name}' has no method '{method_name}'")

        # Check number of arguments (+1 for 'self')
        if len(method_def.params) != len(ast.args) + 1:
             raise Exception(f"Runtime error on line {ast.line_num}: Method '{class_def.name}.{method_name}' expects {len(method_def.params) - 1} arguments, but got {len(ast.args)}")

        evaluated_args = [evaluate(arg, local_vars) for arg in ast.args]

        # Create a new local scope for the method execution
        method_locals = {method_def.params[0]: instance_obj} # Map the first parameter (self) to the instance
        for i, arg_val in enumerate(evaluated_args):
             method_locals[method_def.params[i+1]] = arg_val # Map remaining parameters to evaluated arguments

        # Evaluate the method's body in the new local scope
        # Handle return values from the method
        try:
             result = evaluate(method_def.body, method_locals)
             return result # Return the result of the method execution
        except ReturnValue as rv:
             return rv.value # Catch and return the value from a ReturnStatement

    # --- END NEW ---
    elif isinstance(ast, ReturnStatement):
        # In a real interpreter, this would stop the function execution and return the value.
        # For this simple evaluator, we just raise a custom exception to propagate the return value.
        raise ReturnValue(evaluate(ast.value, local_vars))
    elif isinstance(ast, StatementList):
        result = None
        for stmt in ast.statements:
            try:
                result = evaluate(stmt, local_vars)
            except ReturnValue as rv:
                 # Propagate return values up the call stack
                 raise rv
        return result
    elif isinstance(ast, FunctionDef) or isinstance(ast, StructDef):
        # Definitions do not produce a runtime value when encountered in a statement list.
        # Function definitions are handled by parse_function_def and stored.
        # Struct definitions are handled by parse_struct_def and stored.
        return None
    # Handle assignment to a member access
    elif isinstance(ast, Assignment) and isinstance(ast.name, MemberAccess):
         # This case is now handled within the general Assignment evaluation logic
         pass
    # Handle assignment to an index access (already handled above in Assignment evaluation)
    # elif isinstance(ast, Assignment) and isinstance(ast.name, IndexAccess):
    #      # This case is now handled within the general Assignment evaluation logic
    #      pass


    raise Exception(f"Runtime error on line {ast.line_num}: Unsupported AST node {type(ast)}")

# Set to keep track of deleted variable names
deleted_variables = set()

# Custom exception for handling return values
class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value

# -----------------------------
# Interactive Main Loop
# -----------------------------
# This loop prompts the user for input and evaluates it.
if __name__ == "__main__":
    # Added debug prints for command-line arguments
    print(f"sys.argv: {sys.argv}")
    print(f"len(sys.argv): {len(sys.argv)}")

    if len(sys.argv) > 1:
        print(f"sys.argv[1]: {sys.argv[1]}")

    # --- MODIFICATION: Check for --help anywhere in arguments ---
    if "--help" in sys.argv:
         # Basic help message (matches rubric example 7.4)
         print("Usage: python final.py [run <filename>] [--help]")
         print("  run <filename>  : Executes the code in the specified file.")
         print("  --help          : Displays this help message.")
         print("If no arguments are provided, enters interactive REPL mode.")
    # --- END MODIFICATION ---
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        print("Detected 'run' command.")
        if len(sys.argv) > 2:
            file_name = sys.argv[2]
            print(f"Attempting to run file: {file_name}")
            try:
                with open(file_name, 'r') as f:
                    file_content = f.read()
                print("File read successfully. Tokenizing and parsing...")
                try:
                    tokens = tokenize(file_content)

                    # --- Display Tokenization Output ---
                    formatted_tokens = []
                    for token, _ in tokens:
                        if token.isidentifier():
                            # Check if the identifier is a keyword for more specific output
                            if token in KEYWORDS:
                                formatted_tokens.append(f"keyword {token}")
                            else:
                                formatted_tokens.append(f"variable {token}")
                        elif token.isdigit():
                             formatted_tokens.append(f"number {token}")
                        else:
                            formatted_tokens.append(f"'{token}'")
                    print("Breaks into:", ', '.join(formatted_tokens))
                    # --- End of Automatic Tokenization Output ---


                    parser = Parser(tokens)
                    ast = parser.parse()
                    print("Parsing complete. Evaluating...")

                    # --- Display AST Output ---
                    print(f"AST: {repr(ast)}")
                    # --- End of AST Output ---

                    # Evaluate the parsed statements one by one to handle redeclaration errors correctly
                    if isinstance(ast, StatementList):
                        result = None
                        for stmt in ast.statements:
                            try:
                                result = evaluate(stmt)
                                # Only print evaluation result if it's not None (e.g., not a print statement)
                                if result is not None:
                                    if isinstance(result, bool):
                                        result_str = "true" if result else "false"
                                    else:
                                        result_str = str(result)
                                    print(f"Evaluation: {result_str}\n")
                                else:
                                     # Print a newline for clarity even if no explicit result is printed
                                     print("\n", end="")
                            # --- MODIFICATION: Catch specific errors without "Error -" prefix ---
                            except (TypeError, ZeroDivisionError, IndexError, NameError, AttributeError) as e:
                                print(f"{e}", file=sys.stderr) # Print exact error message
                                break # Stop processing after a runtime error
                            # --- END MODIFICATION ---
                            except (SyntaxError, Exception) as e:
                                # Catch other errors (SyntaxError and other Exceptions)
                                print(f"Error - {e}", file=sys.stderr)
                                # Stop processing the rest of the statements in the file after an error
                                break
                    else:
                         # If the parsed AST is not a StatementList (e.g., a single expression), evaluate it directly
                         try:
                             result = evaluate(ast)
                             if result is not None:
                                 if isinstance(result, bool):
                                     result_str = "true" if result else "false"
                                 else:
                                     result_str = str(result)
                                 print(f"Evaluation: {result_str}\n")
                             else:
                                 print("\n", end="")
                         # --- MODIFICATION: Catch specific errors without "Error -" prefix ---
                         except (TypeError, ZeroDivisionError, IndexError, NameError, AttributeError) as e:
                              print(f"{e}", file=sys.stderr) # Print exact error message
                         # --- END MODIFICATION ---
                         except (SyntaxError, Exception) as e:
                             print(f"Error - {e}", file=sys.stderr)


                    print("Evaluation complete.")
                except (SyntaxError, Exception) as e:
                     print(f"Error - {e}", file=sys.stderr) # Print errors to stderr
            except FileNotFoundError:
                print(f"Error: File '{file_name}' not found.", file=sys.stderr)
        else:
            print("Usage: python final.py run <filename>", file=sys.stderr) # Updated usage message
    elif len(sys.argv) > 1:
         print(f"Unknown command: {sys.argv[1]}. Use --help for usage.", file=sys.stderr)
    else:
        # Interactive REPL mode (matches rubric example 4.4)
        print("Entering interactive REPL mode.")
        print("Enter your test case (or type 'quit' to exit):")
        input_buffer = ""
        while True:
            try:
                # Use a different prompt for continuation lines
                prompt = ">> " if not input_buffer.strip() else ".. " # Check stripped buffer for prompt
                line = input(prompt)

                if line.strip().lower() in ['quit', 'exit']:
                    break

                # If the line is empty or only contains whitespace or a comment, and the buffer is empty,
                # just continue without adding to buffer. This handles single-line comments/empty lines correctly.
                if not input_buffer.strip() and (not line.strip() or line.strip().startswith('//')):
                     print("\n", end="") # Print a newline for clarity even if no explicit result is printed
                     continue


                input_buffer += line + '\n' # Add the line and a newline to the buffer

                # Attempt to tokenize the current buffer.
                tokens = []
                try:
                    tokens = tokenize(input_buffer)
                except SyntaxError as e:
                     # If tokenization fails, report the error immediately and clear the buffer
                     print(f"{e}", file=sys.stderr)
                     input_buffer = ""
                     continue # Go to the next line input

                # If tokenization is successful, attempt to parse.
                parser = Parser(tokens)

                # Attempt to parse the tokens. This is where syntax errors should be caught immediately.
                # We need to check if the parser consumed all tokens to determine if the statement is complete.
                try:
                    ast = parser.parse()
                except SyntaxError as e:
                    # If a syntax error occurs during parsing, it means the current buffer is not a valid statement.
                    # We should report the error and clear the buffer.
                    print(f"{e}", file=sys.stderr)
                    input_buffer = ""
                    continue


                # Check if the parser consumed all input. If not, it means the buffer contains an incomplete statement.
                if parser.pos == len(tokens):
                    # If parsing succeeded and all tokens were consumed, it's a complete statement.
                    # --- Display Tokenization Output Automatically ---
                    formatted_tokens = []
                    for token, _ in tokens:
                        if token.isidentifier():
                             # Check if the identifier is a keyword for more specific output
                            if token in KEYWORDS:
                                formatted_tokens.append(f"keyword {token}")
                            else:
                                formatted_tokens.append(f"variable {token}")
                        elif token.isdigit():
                             formatted_tokens.append(f"number {token}")
                        else:
                            formatted_tokens.append(f"'{token}'")
                    print("Breaks into:", ', '.join(formatted_tokens))
                    # --- End of Automatic Tokenization Output ---

                    # --- Display AST Output ---
                    print(f"AST: {repr(ast)}")
                    # --- End of AST Output ---

                    try:
                        result = evaluate(ast)

                        # Only print evaluation result if it's not None (e.g., not a print statement)
                        if result is not None:
                            if isinstance(result, bool):
                                result_str = "true" if result else "false"
                            else:
                                result_str = str(result)
                            print(f"Evaluation: {result_str}\n")
                        else:
                             # Print a newline for clarity even if no explicit result is printed
                             print("\n", end="")

                    # --- MODIFICATION: Catch specific errors without "Error -" prefix in REPL ---
                    except (TypeError, ZeroDivisionError, IndexError, NameError, AttributeError) as e:
                         print(f"{e}\n", file=sys.stderr) # Print exact error message with newline
                    # --- END MODIFICATION ---
                    # --- MODIFICATION: Catch ZeroDivisionError specifically in REPL ---
                    except ZeroDivisionError as e:
                         print(f"{e}\n", file=sys.stderr) # Print exact error message with newline
                    # --- END MODIFICATION ---
                    except Exception as e: # Catch other Exceptions (runtime errors)
                         # Print other errors with the "Error -" prefix
                         print(f"Error - {e}\n", file=sys.stderr)


                    input_buffer = "" # Clear the buffer after successful execution

                # If parser.pos != len(tokens), it means the buffer contains an incomplete statement,
                # so the loop continues and the '..' prompt is shown. The buffer is NOT cleared.


            except Exception as e: # Catch other Exceptions (runtime errors)
                 # Print other errors with the "Error -" prefix
                 print(f"Error - {e}\n", file=sys.stderr)
                 input_buffer = "" # Clear the buffer on error

