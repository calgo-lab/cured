# error-demo


### Usage 

1. `uv sync`
2. `source .venv/bin/activate`
3. `python3 main.py`

Notes from the chat with Felix:

yeah, it's all ready in my head. 
 
essentially i was thinking we implement a streamlit app with their builtin tables or some nicer version of that like this https://mwouts.github.io/itables/apps/streamlit.html
Streamlit — Interactive Tables
 
there will be like 3 components: 
 
1) table upload or example chosen by us

2) error injection with tab_err (btw, maybe we can still find a better name ... )

3) cleaning with Conformal Data Cleaning

4) detection of error mechanisms and positions, if possible also an explanation of individual errors with XAI on top of CDC (that's an ongoing master's thesis)
