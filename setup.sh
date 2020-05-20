mkdir -p ~/.streamlit/

echo "[general]
email = \"michele.mastropietro@gmail.com\"
" > ~/.streamlit/credentials.toml

echo "[server]
headless = true
enableCORS=false
port = $PORT
" > ~/.streamlit/config.toml

# conda install numpy

# python -m pip install pynbody

python -m pip install git+https://github.com/elehcim/simulation.git
