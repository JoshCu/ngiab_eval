FROM rockylinux:9.1 AS base

# Install dependencies
RUN dnf install python3 python3-pip -y

# Copy the application
COPY . /ngiab_eval

# Install the application
RUN pip3 install /ngiab_eval[plot]

# Run the application
ENTRYPOINT ["python3","-m","ngiab_eval","-i","/ngen/ngen/data"]