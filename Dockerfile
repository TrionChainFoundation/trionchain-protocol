# --- ETAPA 1: CONSTRUCTOR (BUILDER) ---
FROM ubuntu:22.04 as builder

# Evitar preguntas interactivas
ENV DEBIAN_FRONTEND=noninteractive

# 1. Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y \
    git \
    clang \
    curl \
    libssl-dev \
    llvm \
    libudev-dev \
    make \
    protobuf-compiler

# 2. Instalar Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 3. CONFIGURACIÓN CRÍTICA (Aquí estaba el error)
# Aseguramos que la versión estable tenga el target WASM
RUN rustup default stable
RUN rustup update
RUN rustup target add wasm32-unknown-unknown
RUN rustup component add rust-src

# 4. Copiar código
WORKDIR /trionchain
COPY . .

# 5. Compilar
# Nota: Ya no usamos 'cargo update' para respetar tu Cargo.lock local
RUN cargo build --release

# --- ETAPA 2: EJECUTOR (RUNNER) ---
FROM ubuntu:22.04

# Dependencias mínimas para correr
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Copiar el binario
COPY --from=builder /trionchain/target/release/solochain-template-node /usr/local/bin/trionchain-node

# Puertos
EXPOSE 30333 9933 9944 9615

# Comando de inicio
ENTRYPOINT ["/usr/local/bin/trionchain-node"]