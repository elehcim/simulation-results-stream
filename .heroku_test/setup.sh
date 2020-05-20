mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"michele.mastropietro@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

python -m pip install pynbody

python -m pip install git+https://github.com/elehcim/simulation.git
