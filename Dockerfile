# base image
FROM python:3.7

# make working directory of app
WORKDIR .

# copy requirements over
COPY requirements.txt .

# install requirements
RUN pip3 install -r requirements.txt

RUN python -m spacy download en_core_web_sm

# copy files
COPY . .

# expose streamlit port
EXPOSE 8501

# run streamlit
CMD ["streamlit", "run", "app.py"]

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'
