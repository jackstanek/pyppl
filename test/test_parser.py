import pytest

from pyppl import ast as _ast
from pyppl.parser import parse


def parse_expr(src: str) -> _ast.EffectfulNode:
    """Parse a program and return its main expression"""
    prog = parse(src)
    return prog.expr


def test_true_literal():
    """Tests the transformation of a 'true' literal."""
    expr = parse_expr("return true")
    assert isinstance(expr, _ast.ReturnNode)
    assert isinstance(expr.value, _ast.TrueNode)


def test_false_literal():
    """Tests the transformation of a 'false' literal."""
    expr = parse_expr("return false")
    assert isinstance(expr, _ast.ReturnNode)
    assert isinstance(expr.value, _ast.FalseNode)


def test_nil_literal():
    """Tests the transformation of a 'nil' literal."""
    expr = parse_expr("return nil")
    assert isinstance(expr, _ast.ReturnNode)
    assert isinstance(expr.value, _ast.NilNode)


def test_variable_node():
    """Tests the transformation of a variable."""
    expr = parse_expr("return x")
    assert isinstance(expr, _ast.ReturnNode)
    assert isinstance(expr.value, _ast.VariableNode)
    assert expr.value.name == "x"


def test_if_then_else():
    """Tests the transformation of an 'if then else' expression."""
    code = "return if true then x else nil"
    expr = parse_expr(code)
    assert isinstance(expr, _ast.ReturnNode)
    if_node = expr.value
    assert isinstance(if_node, _ast.IfElseNode)
    assert isinstance(if_node.condition, _ast.TrueNode)
    assert isinstance(if_node.true_branch, _ast.VariableNode)
    assert if_node.true_branch.name == "x"
    assert isinstance(if_node.false_branch, _ast.NilNode)


def test_cons_node():
    """Tests the transformation of a 'cons' expression."""
    code = "return cons true false"
    expr = parse_expr(code)
    assert isinstance(expr, _ast.ReturnNode)
    cons_node = expr.value
    assert isinstance(cons_node, _ast.ConsNode)
    assert isinstance(cons_node.head, _ast.TrueNode)
    assert isinstance(cons_node.tail, _ast.FalseNode)


def test_cons_func_app():
    """Tests the transformation of a cons with a function application."""
    code = "return cons (f true) false"
    expr = parse_expr(code)
    assert isinstance(expr, _ast.ReturnNode)
    val = expr.value
    assert isinstance(val, _ast.ConsNode)
    assert isinstance(val.head, _ast.PureApplNode)
    assert isinstance(val.tail, _ast.FalseNode)


def test_func_node():
    """Tests the transformation of an anonymous function expression."""
    code = """define foo = \\x -> true
    return true"""
    fun = parse(code).defns["foo"]
    assert isinstance(fun, _ast.FuncNode)
    assert fun.args == ["x"]
    assert isinstance(fun.body, _ast.TrueNode)


def test_func_binding():
    """Tests the transformation of a function binding."""
    code = """define not x = if x then false else true
    return not true
    """
    prog = parse(code)
    fun = prog.defns["not"]
    assert isinstance(fun, _ast.FuncNode)
    assert fun.args == ["x"]
    assert isinstance(fun.body, _ast.IfElseNode)


def test_func_node_multiple_args():
    """Tests the transformation of an anonymous function expression."""
    code = """define ifthenelse = \\x y z -> if x then y else z
    return true"""
    fun = parse(code).defns["ifthenelse"]
    assert isinstance(fun, _ast.FuncNode)
    assert fun.args == ["x", "y", "z"]
    assert isinstance(fun.body, _ast.IfElseNode)


def test_flip_expr_float():
    """Tests the transformation of a 'flip' expression with a float parameter."""
    expr = parse_expr("flip 0.5")
    assert isinstance(expr, _ast.FlipNode)
    assert expr.theta == 0.5


def test_flip_expr_sym():
    """Tests the transformation of a 'flip' expression with a symbolic parameter."""
    code = "flip p_val"
    expr = parse_expr(code)
    assert isinstance(expr, _ast.FlipNode)
    assert expr.theta == "p_val"  # sym_param returns string, not float


def test_return_expr():
    """Tests the transformation of a 'return' expression."""
    code = "return cons true nil"
    expr = parse_expr(code)
    assert isinstance(expr, _ast.ReturnNode)
    assert isinstance(expr.value, _ast.ConsNode)
    assert isinstance(expr.value.head, _ast.TrueNode)
    assert isinstance(expr.value.tail, _ast.NilNode)


def test_bind_expr_simple():
    """Tests the transformation of a simple bind expression."""
    code = "x <- flip 0.8; return x"
    expr = parse_expr(code)
    assert isinstance(expr, _ast.SequenceNode)
    assert expr.variable_name == "x"
    assert isinstance(expr.assignment_expr, _ast.FlipNode)
    assert expr.assignment_expr.theta == 0.8
    assert isinstance(expr.next_expr, _ast.ReturnNode)
    assert isinstance(expr.next_expr.value, _ast.VariableNode)
    assert expr.next_expr.value.name == "x"


def test_nested_expressions():
    """Tests the transformation of complex nested expressions."""
    code = "a <- flip 0.1; b <- flip 0.9; c <- flip 0.2; return cons (if a then b else c) nil"
    expr = parse_expr(code)

    assert isinstance(expr, _ast.SequenceNode)
    assert expr.variable_name == "a"
    assert isinstance(expr.assignment_expr, _ast.FlipNode)
    assert expr.assignment_expr.theta == 0.1

    # Check the first nested sequence
    nested_seq = expr.next_expr
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


def test_parenthesized_pure_expr():
    """Tests transformation with parentheses around a pure expression."""
    code = "return (true)"
    expr = parse_expr(code)
    assert isinstance(expr, _ast.ReturnNode)
    assert isinstance(expr.value, _ast.TrueNode)


def test_parenthesized_eff_expr():
    """Tests transformation with parentheses around an effectful expression."""
    code = "(return true)"
    expr = parse_expr(code)
    assert isinstance(expr, _ast.ReturnNode)
    assert isinstance(expr.value, _ast.TrueNode)


def test_variable_name_keywords():
    """
    Tests that keywords are NOT parsed as VAR_OR_PARAM_NAME in pure expressions.
    This relies on the Lark grammar correctly preventing this.
    """
    # These should fail to parse if the grammar is correct
    with pytest.raises(Exception):  # Expecting a parse error
        parse("if <- flip 0.5; return if")
    with pytest.raises(Exception):
        parse("return then")
    with pytest.raises(Exception):
        parse("return else")


def test_parsing_program_with_def():
    """
    Test that a program is parsed along with its definitions.
    """
    code = """
        define foo = true
        return foo
    """
    prog = parse(code)
    assert "foo" in prog.defns
