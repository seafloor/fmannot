# FMAnnot

A Python tool to query, amalgamate, and visualize genetic variant annotations from genomic Foundation Models (FMs) and public databases.

This project provides a unified interface for querying variant effects. It currently supports AlphaGenome, with plans to integrate OpenTargets and other tools. The primary goal is to provide a simple lookup and visualization tool (via Streamlit) for researchers with a focus on Alzheimer's disease common variants.

## 📦 Installation

This project is managed with `uv` - install it [here](https://docs.astral.sh/uv/getting-started/installation/).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/seafloor/fmannot
    cd fmannot
    ```

2.  **Create environment and install dependencies:**
    ```bash
    uv venv
    uv sync
    ```

3.  **Set up API keys:**
    Create an `auth.yaml` file in the project root for your API keys. An example file is provided.
    ```bash
    cp example_auth.yaml auth.yaml
    ```
    Now, edit `auth.yaml` with your personal API keys. You get one for alpha genome [here](https://deepmind.google.com/science/alphagenome/).

## 💻 Usage

To browse downloaded data for Alzheimer's disease variants, use the streamlit app:

```bash
streamlit run app/app.py
```

To query databases with your own snps, check out the examples in /notebooks. You can run jupyterlab as:

```bash
uv run --with jupyter jupyter lab
```

## License

This project is licensed under the MIT License.