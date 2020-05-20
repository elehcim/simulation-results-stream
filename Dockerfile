FROM heroku/miniconda

# Grab requirements.txt.
ADD ./webapp/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr /tmp/requirements.txt

# Add our code
ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp

# Python should be there: I use the requirements and then build pynbody...

# RUN conda install numpy
RUN python -m pip install -r requirements
RUN python -m pip install pynbody

RUN python -m pip install git+https://github.com/elehcim/simulation.git

#CMD gunicorn --bind 0.0.0.0:$PORT wsgi
CMD web: sh setup.sh && streamlit run stream.py