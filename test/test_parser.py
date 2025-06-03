import pytest

from pyppl import ast as _ast
from pyppl.parser import PypplTransformer, pyppl_parser


@pytest.fixture
def transformer():
    """Provides a PypplTransformer instance for tests."""
    return PypplTransformer()


@pytest.fixture
def parser():
    """Provides the Pyppl parser instance."""
    return pyppl_parser


def test_true_literal(parser, transformer):
    """Tests the transformation of a 'true' literal."""
    tree = parser.parse("return true")
    ast = transformer.transform(tree)
    assert isinstance(
        ast, _ast.ReturnNode
    )  # 'true' is a pure_expr, wrapped by return_expr by default start rule
    assert isinstance(ast.value, _ast.TrueNode)


def test_false_literal(parser, transformer):
    """Tests the transformation of a 'false' literal."""
    tree = parser.parse("return false")
    ast = transformer.transform(tree)
    assert isinstance(ast, _ast.ReturnNode)
    assert isinstance(ast.value, _ast.FalseNode)


def test_nil_literal(parser, transformer):
    """Tests the transformation of a 'nil' literal."""
    tree = parser.parse("return nil")
    ast = transformer.transform(tree)
    assert isinstance(ast, _ast.ReturnNode)
    assert isinstance(ast.value, _ast.NilNode)


def test_variable_node(parser, transformer):
    """Tests the transformation of a variable."""
    tree = parser.parse("return x")
    ast = transformer.transform(tree)
    assert isinstance(ast, _ast.ReturnNode)
    assert isinstance(ast.value, _ast.VariableNode)
    assert ast.value.name == "x"


def test_if_then_else(parser, transformer):
    """Tests the transformation of an 'if then else' expression."""
    code = "return if true then x else nil"
    tree = parser.parse(code)
    ast = transformer.transform(tree)
    assert isinstance(ast, _ast.ReturnNode)
    if_node = ast.value
    assert isinstance(if_node, _ast.IfElseNode)
    assert isinstance(if_node.condition, _ast.TrueNode)
    assert isinstance(if_node.true_branch, _ast.VariableNode)
    assert if_node.true_branch.name == "x"
    assert isinstance(if_node.false_branch, _ast.NilNode)


def test_cons_node(parser, transformer):
    """Tests the transformation of a 'cons' expression."""
    code = "return cons true false"
    tree = parser.parse(code)
    ast = transformer.transform(tree)
    assert isinstance(ast, _ast.ReturnNode)
    cons_node = ast.value
    assert isinstance(cons_node, _ast.ConsNode)
    assert isinstance(cons_node.head, _ast.TrueNode)
    assert isinstance(cons_node.tail, _ast.FalseNode)


def test_flip_expr_float(parser, transformer):
    """Tests the transformation of a 'flip' expression with a float parameter."""
    code = "flip 0.5"
    tree = parser.parse(code)
    ast = transformer.transform(tree)
    assert isinstance(ast, _ast.FlipNode)
    assert ast.theta == 0.5


def test_flip_expr_sym(parser, transformer):
    """Tests the transformation of a 'flip' expression with a symbolic parameter."""
    code = "flip p_val"
    tree = parser.parse(code)
    ast = transformer.transform(tree)
    assert isinstance(ast, _ast.FlipNode)
    assert ast.theta == "p_val"  # sym_param returns string, not float


def test_return_expr(parser, transformer):
    """Tests the transformation of a 'return' expression."""
    code = "return cons true nil"
    tree = parser.parse(code)
    ast = transformer.transform(tree)
    assert isinstance(ast, _ast.ReturnNode)
    assert isinstance(ast.value, _ast.ConsNode)
    assert isinstance(ast.value.head, _ast.TrueNode)
    assert isinstance(ast.value.tail, _ast.NilNode)


def test_bind_expr_simple(parser, transformer):
    """Tests the transformation of a simple bind expression."""
    code = "x <- flip 0.8; return x"
    tree = parser.parse(code)
    ast = transformer.transform(tree)
    assert isinstance(ast, _ast.SequenceNode)
    assert ast.variable_name == "x"
    assert isinstance(ast.assignment_expr, _ast.FlipNode)
    assert ast.assignment_expr.theta == 0.8
    assert isinstance(ast.next_expr, _ast.ReturnNode)
    assert isinstance(ast.next_expr.value, _ast.VariableNode)
    assert ast.next_expr.value.name == "x"


def test_nested_expressions(parser, transformer):
    """Tests the transformation of complex nested expressions."""
    code = "a <- flip 0.1; b <- flip 0.9; c <- flip 0.2; return cons (if a then b else c) nil"
    tree = parser.parse(code)
    ast = transformer.transform(tree)

    assert isinstance(ast, _ast.SequenceNode)
    assert ast.variable_name == "a"
    assert isinstance(ast.assignment_expr, _ast.FlipNode)
    assert ast.assignment_expr.theta == 0.1

    # Check the first nested sequence
    nested_seq = ast.next_expr
    assert isinstance(nested_seq, _ast.SequenceNode)
    assert nested_seq.variable_name == "b"

    # Check the second nested sequence
    nested_seq = nested_seq.next_expr
    assert isinstance(nested_seq, _ast.SequenceNode)
    assert nested_seq.variable_name == "c"

    # Check the final return
    final_return = nested_seq.next_expr
    assert isinstance(final_return, _ast.ReturnNode)
    cons_node = final_return.value
    assert isinstance(cons_node, _ast.ConsNode)
    head_node = cons_node.head
    assert isinstance(head_node, _ast.IfElseNode)
    assert isinstance(head_node.condition, _ast.VariableNode)
    assert head_node.condition.name == "a"
    assert isinstance(head_node.true_branch, _ast.VariableNode)
    assert head_node.true_branch.name == "b"
    assert isinstance(head_node.false_branch, _ast.VariableNode)
    assert head_node.false_branch.name == "c"
    assert isinstance(cons_node.tail, _ast.NilNode)


def test_parenthesized_pure_expr(parser, transformer):
    """Tests transformation with parentheses around a pure expression."""
    code = "return (true)"
    tree = parser.parse(code)
    ast = transformer.transform(tree)
    assert isinstance(ast, _ast.ReturnNode)
    assert isinstance(ast.value, _ast.TrueNode)


def test_parenthesized_eff_expr(parser, transformer):
    """Tests transformation with parentheses around an effectful expression."""
    code = "(return true)"
    tree = parser.parse(code)
    ast = transformer.transform(tree)
    assert isinstance(ast, _ast.ReturnNode)
    assert isinstance(ast.value, _ast.TrueNode)


def test_variable_name_keywords(parser, transformer):
    """
    Tests that keywords are NOT parsed as VAR_OR_PARAM_NAME in pure expressions.
    This relies on the Lark grammar correctly preventing this.
    """
    # These should fail to parse if the grammar is correct
    with pytest.raises(Exception):  # Expecting a parse error
        parser.parse("if <- flip 0.5; return if")
    with pytest.raises(Exception):
        parser.parse("return then")
    with pytest.raises(Exception):
        parser.parse("return else")
