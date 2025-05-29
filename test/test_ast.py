import pytest

from pyppl.ast import (
    ConsNode,
    Environment,
    ExpressionNode,
    FalseNode,
    FlipNode,
    IfElseNode,
    NilNode,
    PureNode,
    ReturnNode,
    SequenceNode,
    TrueNode,
    VariableNode,
    var,
)


# Assume PureNode is defined elsewhere, for testing purposes we'll mock it.
class MockPureNode(PureNode):
    """A simple mock class to stand in for PureNode for testing."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, MockPureNode):
            return NotImplemented
        return self.value == other.value

    def __repr__(self):
        return f"MockPureNode({self.value})"


# Pytest functions to replace unittest.TestCase methods
def test_initialization():
    """Test that the Environment is initialized with one empty scope."""
    env = Environment()
    assert len(env.scopes) == 1
    assert env.scopes[0] == {}


def test_add_scope():
    """Test adding a new scope to the environment."""
    env = Environment()
    initial_scopes_len = len(env.scopes)
    env.add_scope()
    assert len(env.scopes) == initial_scopes_len + 1
    assert env.scopes[-1] == {}  # New scope should be empty


def test_remove_scope():
    """Test removing a scope from the environment."""
    env = Environment()
    env.add_scope()  # Add a second scope to remove
    initial_scopes_len = len(env.scopes)
    env.remove_scope()
    assert len(env.scopes) == initial_scopes_len - 1
    assert env.scopes[-1] == {}  # Should be back to the initial empty scope


def test_remove_last_scope():
    """
    Test removing the very last scope.
    Note: The current implementation allows this, which might be an
    undesired state for an environment that always needs a global scope.
    """
    env = Environment()
    # Initial state: [{'global_scope'}]
    assert len(env.scopes) == 1
    env.remove_scope()
    # After removing, scopes should be empty: []
    assert len(env.scopes) == 0
    # Attempting to add a binding after removing the last scope would cause IndexError
    with pytest.raises(IndexError):
        env.add_binding("x", MockPureNode(10))


def test_add_binding_new_name():
    """Test adding a new binding to the current local scope."""
    env = Environment()
    node_x = MockPureNode(10)
    env.add_binding("x", node_x)
    assert env.scopes[-1]["x"] == node_x


def test_add_binding_existing_name_local_scope():
    """
    Test adding a binding with a name that already exists in the local scope
    should raise a ValueError.
    """
    env = Environment()
    node_x1 = MockPureNode(10)
    node_x2 = MockPureNode(20)
    env.add_binding("x", node_x1)
    with pytest.raises(ValueError, match="name x already bound in local scope"):
        env.add_binding("x", node_x2)


def test_get_binding_local_scope():
    """Test retrieving a binding from the local scope."""
    env = Environment()
    node_y = MockPureNode("hello")
    env.add_binding("y", node_y)
    assert env.get_binding("y") == node_y


def test_get_binding_outer_scope():
    """Test retrieving a binding from an outer scope."""
    env = Environment()
    node_a = MockPureNode(100)
    env.add_binding("a", node_a)  # Add to global scope

    env.add_scope()  # Add a new scope
    node_b = MockPureNode(200)
    env.add_binding("b", node_b)  # Add to new scope

    assert env.get_binding("a") == node_a  # 'a' should be found in outer scope
    assert env.get_binding("b") == node_b  # 'b' should be found in local scope


def test_get_binding_shadowing():
    """Test binding lookup with shadowing (local scope hides outer scope)."""
    env = Environment()
    node_x_outer = MockPureNode(10)
    env.add_binding("x", node_x_outer)  # x in global scope

    env.add_scope()  # New scope
    node_x_inner = MockPureNode(20)
    env.add_binding("x", node_x_inner)  # x in local scope (shadows outer x)

    assert env.get_binding("x") == node_x_inner  # Should get the inner 'x'

    env.remove_scope()  # Remove inner scope
    assert env.get_binding("x") == node_x_outer  # Should now get the outer 'x'


def test_get_binding_not_found():
    """Test that attempting to get a non-existent binding raises a ValueError."""
    env = Environment()
    with pytest.raises(ValueError, match="name z not bound"):
        env.get_binding("z")

    env.add_scope()
    env.add_binding("foo", MockPureNode("bar"))
    with pytest.raises(ValueError, match="name baz not bound"):
        env.get_binding("baz")


def test_complex_scope_management_and_bindings():
    """Test a more complex scenario with multiple scopes and bindings."""
    env = Environment()  # scopes = [{}]
    node1 = MockPureNode(1)
    env.add_binding("var1", node1)  # scopes = [{'var1': 1}]

    env.add_scope()  # scopes = [{'var1': 1}, {}]
    node2 = MockPureNode(2)
    env.add_binding("var2", node2)  # scopes = [{'var1': 1}, {'var2': 2}]
    node3 = MockPureNode(3)
    env.add_binding("var3", node3)  # scopes = [{'var1': 1}, {'var2': 2, 'var3': 3}]

    assert env.get_binding("var1") == node1
    assert env.get_binding("var2") == node2
    assert env.get_binding("var3") == node3

    env.add_scope()  # scopes = [{'var1': 1}, {'var2': 2, 'var3': 3}, {}]
    node1_shadow = MockPureNode(10)
    env.add_binding(
        "var1", node1_shadow
    )  # scopes = [{'var1': 1}, {'var2': 2, 'var3': 3}, {'var1': 10}]
    node4 = MockPureNode(4)
    env.add_binding(
        "var4", node4
    )  # scopes = [{'var1': 1}, {'var2': 2, 'var3': 3}, {'var1': 10, 'var4': 4}]

    assert env.get_binding("var1") == node1_shadow  # Shadowed
    assert env.get_binding("var2") == node2
    assert env.get_binding("var3") == node3
    assert env.get_binding("var4") == node4

    env.remove_scope()  # scopes = [{'var1': 1}, {'var2': 2, 'var3': 3}]
    assert env.get_binding("var1") == node1  # Back to original var1
    assert env.get_binding("var2") == node2
    assert env.get_binding("var3") == node3
    with pytest.raises(ValueError, match="name var4 not bound"):
        env.get_binding("var4")

    env.remove_scope()  # scopes = [{'var1': 1}]
    with pytest.raises(ValueError, match="name var2 not bound"):
        env.get_binding("var2")
    with pytest.raises(ValueError, match="name var3 not bound"):
        env.get_binding("var3")
    assert env.get_binding("var1") == node1


def test_purenode_eval():
    """Test PureNode.eval returns self."""
    node = MockPureNode("hello")
    env = Environment()
    assert node.eval(env) is node


def test_variablenode_eval():
    """Test VariableNode.eval retrieves binding from environment."""
    env = Environment()
    var_name = "my_var"
    bound_value = TrueNode()
    env.add_binding(var_name, bound_value)

    var_node = VariableNode(var_name)
    assert var_node.eval(env) is bound_value

    # Test for unbound variable
    unbound_var_node = VariableNode("non_existent_var")
    with pytest.raises(ValueError, match="name non_existent_var not bound"):
        unbound_var_node.eval(env)


def test_ifelsenode_eval_true_condition():
    """Test IfElseNode.eval with a true condition."""
    env = Environment()
    condition = TrueNode()
    true_branch_result = NilNode()
    false_branch_result = VariableNode("x")  # This won't be evaluated

    # Mock eval methods of branches to control their return values
    class MockTrueBranch(PureNode):
        def eval(self, env):
            return true_branch_result

    class MockFalseBranch(PureNode):
        def eval(self, env):
            return false_branch_result

    if_else_node = IfElseNode(condition, MockTrueBranch(), MockFalseBranch())
    assert if_else_node.eval(env) is true_branch_result


def test_ifelsenode_eval_false_condition():
    """Test IfElseNode.eval with a false condition."""
    env = Environment()
    condition = FalseNode()
    true_branch_result = NilNode()  # This won't be evaluated
    false_branch_result = VariableNode("x")

    class MockTrueBranch(PureNode):
        def eval(self, env):
            return true_branch_result

    class MockFalseBranch(PureNode):
        def eval(self, env):
            return false_branch_result

    if_else_node = IfElseNode(condition, MockTrueBranch(), MockFalseBranch())
    assert if_else_node.eval(env) is false_branch_result


def test_consonde_eval():
    """Test ConsNode.eval evaluates head and tail and returns a new ConsNode."""
    env = Environment()
    head_val = TrueNode()
    tail_val = NilNode()

    # Create mock nodes that return specific values when evaluated
    class MockHead(PureNode):
        def eval(self, env):
            return head_val

    class MockTail(PureNode):
        def eval(self, env):
            return tail_val

    cons_node = ConsNode(MockHead(), MockTail())
    result_cons = cons_node.eval(env)

    assert isinstance(result_cons, ConsNode)
    assert result_cons.head is head_val
    assert result_cons.tail is tail_val
    assert result_cons is not cons_node  # Should be a new instance


# --- Pytest Tests for sample() methods ---


def test_returnnode_sample():
    """Test ReturnNode.sample returns the wrapped value."""
    env = Environment()
    value_node = TrueNode()
    return_node = ReturnNode(value_node)
    assert return_node.sample(env) is value_node


def test_returnnode_sample_var():
    """Test ReturnNode.sample returns a variable lookup."""
    env = Environment()
    value_node = TrueNode()
    varname = "x"
    var_node = var(varname)
    env.add_binding(varname, value_node)
    return_node = ReturnNode(var_node)
    assert return_node.sample(env) is value_node


def test_flipnode_sample_true(mocker):
    """Test FlipNode.sample returns TrueNode when random.random() is less than theta."""
    env = Environment()
    theta = 0.7
    flip_node = FlipNode(theta)

    # Mock random.random() to return a value that makes the condition true
    mocker.patch("random.random", return_value=0.5)
    result = flip_node.sample(env)
    assert isinstance(result, TrueNode)


def test_flipnode_sample_false(mocker):
    """Test FlipNode.sample returns FalseNode when random.random() is greater than or equal to theta."""
    env = Environment()
    theta = 0.3
    flip_node = FlipNode(theta)

    # Mock random.random() to return a value that makes the condition false
    mocker.patch("random.random", return_value=0.5)
    result = flip_node.sample(env)
    assert isinstance(result, FalseNode)


def test_sequencenode_sample(mocker):
    """Test SequenceNode.sample binds assignment_expr result and samples next_expr."""
    env = Environment()
    var_name = "my_seq_var"
    assigned_value = NilNode()
    next_expr_result = TrueNode()

    # Mock assignment_expr.sample to return a specific value
    mock_assignment_expr = mocker.Mock(spec=ExpressionNode)
    mock_assignment_expr.sample.return_value = assigned_value

    # Mock next_expr.sample to return a specific value
    mock_next_expr = mocker.Mock(spec=ExpressionNode)
    mock_next_expr.sample.return_value = next_expr_result

    # Spy on env.add_binding to check if it's called correctly
    mocker.spy(env, "add_binding")

    sequence_node = SequenceNode(var_name, mock_assignment_expr, mock_next_expr)
    result = sequence_node.sample(env)

    # Assertions
    mock_assignment_expr.sample.assert_called_once_with(env)
    env.add_binding.assert_called_once_with(var_name, assigned_value)  # type: ignore
    mock_next_expr.sample.assert_called_once_with(env)
    assert result is next_expr_result

    # Verify the binding was actually added to the environment
    assert env.get_binding(var_name) is assigned_value
