FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    gfortran \
    make \
    liblapacke-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Remove pre-compiled binaries committed to the repo (built for x86-64,
# incompatible with arm64 and any other non-matching target architecture)
RUN find . -name "*.o" -delete && find . -name "*.a" -delete

# Build bundled iniparser library
RUN cd vendor/iniparser && make

# Build bundled wcstools library
RUN cd vendor/wcstools-3.9.2/libwcs && make

# Build the adam binary
RUN make adam

RUN mkdir -p /app/data/output

ENTRYPOINT ["/app/adam"]
