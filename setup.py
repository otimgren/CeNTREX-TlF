import setuptools

setuptools.setup(
    name="CeNTREX_TlF",
    author="Olivier Grasdijk",
    author_email="olivier.grasdijk@yale.edu",
    description="general utility package for TlF molecular calculations used in the CeNTREX experiment",
    url="https://github.com/",
    packages=setuptools.find_packages(),
    install_requires=["numpy","scipy", "sympy>=1.9", "tqdm", "rich"],
    data_files=[
        (
            "centrex_TlF/pre_calculated",
            [
                "centrex_TlF/pre_calculated/transformation.db",
                "centrex_TlF/pre_calculated/uncoupled_hamiltonian_X.db",
                "centrex_TlF/pre_calculated/coupled_hamiltonian_B.db",
                "centrex_TlF/pre_calculated/matrix_elements.db",
                "centrex_TlF/pre_calculated/precalculated.json",
                "centrex_TlF/pre_calculated/transitions.pickle",
            ],
        ),
    ],
    python_requires=">=3.6",
    version="0.1",
)
