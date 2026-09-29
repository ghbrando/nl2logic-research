FROM ghcr.io/charmbracelet/vhs
RUN apt-get update && apt-get install -y --no-install-recommends python3 unzip curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -sL https://github.com/vprover/vampire/releases/download/v5.0.1/vampire-Linux-X64.zip -o /tmp/v.zip \
 && unzip -q /tmp/v.zip -d /opt/vampire && find /opt/vampire -type f -name 'vampire*' -exec chmod +x {} \; \
 && ln -s "$(find /opt/vampire -type f -name 'vampire*' | head -1)" /usr/local/bin/vampire && rm /tmp/v.zip
RUN ln -sf /usr/bin/python3 /usr/local/bin/python
