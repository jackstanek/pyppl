import pytest

from pyppl import ast
from pyppl.analysis import NameEnv, UnboundNameError, name_analysis
from pyppl.ast import EvalEnv
from pyppl.params import ParamVector


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

    def test_complex_program_with_multiple_definitions_and_sequences(self):
        # defns: { "g": ff }
        # expr: x <- return g; y <- return tt; return (if y then x else z)
        # 'z' is unbound
        prog = ast.Program(
            defns={"g": ast.FalseNode()},
            expr=ast.SequenceNode(
                variable_name="x",
                assignment_expr=ast.ReturnNode(ast.VariableNode("g")),
                next_expr=ast.SequenceNode(
                    variable_name="y",
                    assignment_expr=ast.ReturnNode(ast.TrueNode()),
                    next_expr=ast.ReturnNode(
                        ast.IfElseNode(
                            condition=ast.VariableNode("y"),
                            true_branch=ast.VariableNode("x"),
                            false_branch=ast.VariableNode("z"),  # unbound
                        )
                    ),
                ),
            ),
        )
        with pytest.raises(UnboundNameError) as excinfo:
            name_analysis(prog)
        assert excinfo.value.name == "z"

    def test_cons_node_with_variables(self):
        # defns: {"h": tt, "t": ff}
        # expr: return (cons h t)
        prog = ast.Program(
            defns={"h": ast.TrueNode(), "t": ast.FalseNode()},
            expr=ast.ReturnNode(
                ast.ConsNode(ast.VariableNode("h"), ast.VariableNode("t"))
            ),
        )
        try:
            name_analysis(prog)
        except UnboundNameError as e:
            pytest.fail(f"Name analysis raised UnboundNameError unexpectedly: {e}")

    def test_infer_pure_node(self):
        env = EvalEnv(params=ParamVector())
        true_node = ast.TrueNode()
        false_node = ast.FalseNode()

        assert true_node.infer(env, ast.TrueNode()) == 1.0
        assert true_node.infer(env, ast.FalseNode()) == 0.0
        assert false_node.infer(env, ast.FalseNode()) == 1.0
        assert false_node.infer(env, ast.TrueNode()) == 0.0

    def test_infer_flip_node(self):
        env = EvalEnv(params=ParamVector({"p1": 0.3}))
        flip_node = ast.FlipNode(theta="p1")

        assert flip_node.infer(env, ast.TrueNode()) == 0.3
        assert flip_node.infer(env, ast.FalseNode()) == 0.7

        flip_node_fixed = ast.FlipNode(theta=0.6)
        assert flip_node_fixed.infer(env, ast.TrueNode()) == 0.6
        assert flip_node_fixed.infer(env, ast.FalseNode()) == 0.4

    def test_infer_return_node(self):
        env = EvalEnv(params=ParamVector())
        return_true = ast.ReturnNode(ast.TrueNode())
        return_false = ast.ReturnNode(ast.FalseNode())

        assert return_true.infer(env, ast.TrueNode()) == 1.0
        assert return_true.infer(env, ast.FalseNode()) == 0.0
        assert return_false.infer(env, ast.FalseNode()) == 1.0
        assert return_false.infer(env, ast.TrueNode()) == 0.0

    def test_infer_sequence_node(self):
        # x <- flip 0.5; return x
        # P(x=tt) = 0.5, P(x=ff) = 0.5
        # return x
        # P(result=tt) = 0.5, P(result=ff) = 0.5
        env = EvalEnv(params=ParamVector())
        seq_node = ast.SequenceNode(
            variable_name="x",
            assignment_expr=ast.FlipNode(theta=0.5),
            next_expr=ast.ReturnNode(ast.VariableNode("x")),
        )

        assert seq_node.infer(env, ast.TrueNode()) == pytest.approx(0.5)
        assert seq_node.infer(env, ast.FalseNode()) == pytest.approx(0.5)

        # y <- flip 0.3; x <- flip 0.7; return (if y then x else ff)
        # P(result=tt) = P(y=tt and x=tt) = 0.3 * 0.7 = 0.21
        # P(result=ff) = P(y=tt and x=ff) + P(y=ff) = 0.3 * 0.3 + 0.7 = 0.09 + 0.7 = 0.79
        env_complex = EvalEnv(params=ParamVector())
        complex_seq = ast.SequenceNode(
            variable_name="y",
            assignment_expr=ast.FlipNode(theta=0.3),
            next_expr=ast.SequenceNode(
                variable_name="x",
                assignment_expr=ast.FlipNode(theta=0.7),
                next_expr=ast.ReturnNode(
                    ast.IfElseNode(
                        condition=ast.VariableNode("y"),
                        true_branch=ast.VariableNode("x"),
                        false_branch=ast.FalseNode(),
                    )
                ),
            ),
        )
        assert complex_seq.infer(env_complex, ast.TrueNode()) == pytest.approx(0.21)
        assert complex_seq.infer(env_complex, ast.FalseNode()) == pytest.approx(0.79)

    def test_gradient_flip_node(self):
        env = EvalEnv(params=ParamVector({"p": 0.5}))
        flip_p = ast.FlipNode(theta="p")

        # Derivative wrt 'p' for ast.TrueNode: d/dp (p) = 1.0
        assert flip_p.deriv(env, "p", ast.TrueNode()) == pytest.approx(1.0)
        # Derivative wrt 'p' for ast.FalseNode: d/dp (1-p) = -1.0
        assert flip_p.deriv(env, "p", ast.FalseNode()) == pytest.approx(-1.0)

        # Derivative wrt a non-existent parameter
        assert flip_p.deriv(env, "q", ast.TrueNode()) == pytest.approx(0.0)

        # Derivative for a fixed theta (should be 0)
        flip_fixed = ast.FlipNode(theta=0.5)
        assert flip_fixed.deriv(env, "p", ast.TrueNode()) == pytest.approx(0.0)

    def test_gradient_sequence_node(self):
        # ast.Program: x <- flip p; return x
        # Denotation for True: p
        # Gradient wrt p for True: 1.0
        # Denotation for False: 1-p
        # Gradient wrt p for False: -1.0
        env = EvalEnv(params=ParamVector({"p": 0.5}))
        seq_node = ast.SequenceNode(
            variable_name="x",
            assignment_expr=ast.FlipNode(theta="p"),
            next_expr=ast.ReturnNode(ast.VariableNode("x")),
        )

        assert seq_node.deriv(env, "p", ast.TrueNode()) == pytest.approx(1.0)
        assert seq_node.deriv(env, "p", ast.FalseNode()) == pytest.approx(-1.0)

        # ast.Program: x <- flip p; y <- flip q; return (if x then y else ff)
        # Denotation for True: P(x=tt, y=tt) = p * q
        # Denotation for False: P(x=tt, y=ff) + P(x=ff) = p * (1-q) + (1-p)
        # Gradient wrt p for True: d/dp (p*q) = q
        # Gradient wrt q for True: d/dq (p*q) = p
        # Gradient wrt p for False: d/dp (p*(1-q) + (1-p)) = (1-q) - 1 = -q
        # Gradient wrt q for False: d/dq (p*(1-q) + (1-p)) = -p

        env_nested = EvalEnv(params=ParamVector({"p": 0.5, "q": 0.2}))
        nested_seq = ast.SequenceNode(
            variable_name="x",
            assignment_expr=ast.FlipNode(theta="p"),
            next_expr=ast.SequenceNode(
                variable_name="y",
                assignment_expr=ast.FlipNode(theta="q"),
                next_expr=ast.ReturnNode(
                    ast.IfElseNode(
                        condition=ast.VariableNode("x"),
                        true_branch=ast.VariableNode("y"),
                        false_branch=ast.FalseNode(),
                    )
                ),
            ),
        )

        # Test gradient wrt 'p' for ast.TrueNode
        assert nested_seq.deriv(env_nested, "p", ast.TrueNode()) == pytest.approx(
            env_nested.get_param("q")
        )
        # Test gradient wrt 'q' for ast.TrueNode
        assert nested_seq.deriv(env_nested, "q", ast.TrueNode()) == pytest.approx(
            env_nested.get_param("p")
        )

        # Test gradient wrt 'p' for ast.FalseNode
        assert nested_seq.deriv(env_nested, "p", ast.FalseNode()) == pytest.approx(
            -env_nested.get_param("q")
        )
        # Test gradient wrt 'q' for ast.FalseNode
        assert nested_seq.deriv(env_nested, "q", ast.FalseNode()) == pytest.approx(
            -env_nested.get_param("p")
        )
