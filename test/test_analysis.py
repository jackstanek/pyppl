import pytest

from pyppl import ast
from pyppl.analysis import NameEnv, UnboundNameError, name_analysis


class TestNameAnalysisPytest:
    def test_name_env_add_and_check(self):
        env = NameEnv()
        assert env.scopes == [set()]
        env.add_name("x")
        assert env.scopes == [{"x"}]
        env.check_binding("x")
        with pytest.raises(UnboundNameError):
            env.check_binding("y")

    def test_name_env_scopes(self):
        env = NameEnv()
        env.add_name("global_var")
        assert env.scopes == [{"global_var"}]

        with env.local_scope():
            env.add_name("local_var")
            assert env.scopes == [{"global_var"}, {"local_var"}]
            env.check_binding("global_var")
            env.check_binding("local_var")
            with pytest.raises(UnboundNameError):
                env.check_binding("another_var")

        assert env.scopes == [{"global_var"}]
        env.check_binding("global_var")
        with pytest.raises(UnboundNameError):
            env.check_binding("local_var")

    def test_valid_program_no_definitions(self):
        # return tt
        prog = ast.Program(defns={}, expr=ast.ReturnNode(value=ast.TrueNode()))
        try:
            name_analysis(prog)
        except UnboundNameError as e:
            pytest.fail(f"Name analysis raised UnboundNameError unexpectedly: {e}")

    def test_valid_program_with_definition(self):
        # defns: { "x": tt }
        # expr: return x
        prog = ast.Program(
            defns={"x": ast.TrueNode()},
            expr=ast.ReturnNode(value=ast.VariableNode("x")),
        )
        try:
            name_analysis(prog)
        except UnboundNameError as e:
            pytest.fail(f"Name analysis raised UnboundNameError unexpectedly: {e}")

    def test_unbound_variable_in_main_expr(self):
        # return y (where y is not defined)
        prog = ast.Program(defns={}, expr=ast.ReturnNode(value=ast.VariableNode("y")))
        with pytest.raises(UnboundNameError) as excinfo:
            name_analysis(prog)
        assert excinfo.value.name == "y"

    def test_unbound_variable_in_definition(self):
        # defns: { "x": y } (where y is not defined)
        # expr: return x
        prog = ast.Program(
            defns={"x": ast.VariableNode("y")},
            expr=ast.ReturnNode(value=ast.VariableNode("x")),
        )
        with pytest.raises(UnboundNameError) as excinfo:
            name_analysis(prog)
        assert excinfo.value.name == "y"

    def test_sequence_node_binding(self):
        # x <- return tt; return x
        prog = ast.Program(
            defns={},
            expr=ast.SequenceNode(
                variable_name="x",
                assignment_expr=ast.ReturnNode(ast.TrueNode()),
                next_expr=ast.ReturnNode(ast.VariableNode("x")),
            ),
        )
        try:
            name_analysis(prog)
        except UnboundNameError as e:
            pytest.fail(f"Name analysis raised UnboundNameError unexpectedly: {e}")

    def test_sequence_node_binding_unbound_in_assignment(self):
        # x <- return y; return x (y is unbound)
        prog = ast.Program(
            defns={},
            expr=ast.SequenceNode(
                variable_name="x",
                assignment_expr=ast.ReturnNode(ast.VariableNode("y")),  # y is unbound
                next_expr=ast.ReturnNode(ast.VariableNode("x")),
            ),
        )
        with pytest.raises(UnboundNameError) as excinfo:
            name_analysis(prog)
        assert excinfo.value.name == "y"

    def test_sequence_node_binding_unbound_in_next_expr(self):
        # x <- return tt; return y (y is unbound)
        prog = ast.Program(
            defns={},
            expr=ast.SequenceNode(
                variable_name="x",
                assignment_expr=ast.ReturnNode(ast.TrueNode()),
                next_expr=ast.ReturnNode(ast.VariableNode("y")),  # y is unbound
            ),
        )
        with pytest.raises(UnboundNameError) as excinfo:
            name_analysis(prog)
        assert excinfo.value.name == "y"

    def test_shadowing_in_sequence(self):
        # defns: { "x": ff }
        # expr: x <- return tt; return x
        # This tests that the inner 'x' shadows the global 'x'.
        # The inner 'x' should be bound by the sequence.
        prog = ast.Program(
            defns={"x": ast.FalseNode()},
            expr=ast.SequenceNode(
                variable_name="x",
                assignment_expr=ast.ReturnNode(ast.TrueNode()),
                next_expr=ast.ReturnNode(ast.VariableNode("x")),
            ),
        )
        try:
            name_analysis(prog)
        except UnboundNameError as e:
            pytest.fail(f"Name analysis raised UnboundNameError unexpectedly: {e}")

    def test_if_else_branch_scoping(self):
        # if tt then return x else return y (x, y unbound)
        prog = ast.Program(
            defns={},
            expr=ast.ReturnNode(
                ast.IfElseNode(
                    condition=ast.TrueNode(),
                    true_branch=ast.VariableNode("x"),
                    false_branch=ast.VariableNode("y"),
                )
            ),
        )
        with pytest.raises(UnboundNameError) as excinfo:
            name_analysis(prog)
        # The error could be 'x' or 'y' depending on traversal order.
        assert excinfo.value.name in ["x", "y"]

    def test_nested_sequence_binding(self):
        # a <- return tt; b <- return a; return b
        prog = ast.Program(
            defns={},
            expr=ast.SequenceNode(
                variable_name="a",
                assignment_expr=ast.ReturnNode(ast.TrueNode()),
                next_expr=ast.SequenceNode(
                    variable_name="b",
                    assignment_expr=ast.ReturnNode(ast.VariableNode("a")),
                    next_expr=ast.ReturnNode(ast.VariableNode("b")),
                ),
            ),
        )
        try:
            name_analysis(prog)
        except UnboundNameError as e:
            pytest.fail(f"Name analysis raised UnboundNameError unexpectedly: {e}")

    def test_mutual_recursive_reference(self):
        """Test that "define" syntax can mutually reference other variables"""
        prog = ast.Program(
            defns={"x": ast.TrueNode(), "y": ast.var("x")},
            expr=ast.ReturnNode(ast.var("y")),
        )
        name_analysis(prog)

    def test_dynamic_scoping_fails(self):
        """Test that dynamic scoping is not accepted"""
        prog = ast.Program(
            defns={"x": ast.var("y")},
            expr=ast.SequenceNode(
                "y", ast.ReturnNode(ast.TrueNode()), ast.ReturnNode(ast.var("x"))
            ),
        )
        with pytest.raises(UnboundNameError):
            name_analysis(prog)
