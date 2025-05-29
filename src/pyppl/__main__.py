from pyppl import ast


def main():
    prog = ast.SequenceNode(
        "x",
        ast.FlipNode(0.5),
        ast.ReturnNode(
            ast.IfElseNode(
                ast.var("x"), ast.ConsNode(ast.var("x"), ast.NilNode()), ast.NilNode()
            )
        ),
    )

    print(prog.sample_toplevel(k=10))


if __name__ == "__main__":
    main()
