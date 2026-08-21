# FaceRecSS — Proteção de Ecrã com Reconhecimento Facial & Registo Simples

## 🚀 TL;DR

O **FaceRecSS** é um bloqueador de ecrã/screensaver interativo escrito em Python.
Deteta movimento através da webcam, reconhece rostos na base de dados, apresenta saudações personalizadas no ecrã, regista eventos em `log.txt` e bloqueia atalhos do sistema (Alt+Tab, Super key) até ser introduzida a palavra-passe secreta.


```bash
# Instalação rápida e execução:
make install   # Instala dependências Python
make models    # Descarrega modelos ONNX para reconhecimento facial
make run       # Executa o FaceRecSS
```

---

## 📋 Índice
- [TL;DR](#-tldr)
- [Funcionalidades Principais](#-funcionalidades-principais)
- [Arquitetura & Como Funciona](#-arquitetura--como-funciona)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Requisitos do Sistema](#-requisitos-do-sistema)
- [Instalação](#-instalação)
- [Configuração (`config.json`)](#-configuração-configjson)
- [Cadastrar Pessoas (`faces/`)](#-cadastrar-pessoas-faces)
- [Registos (`log.txt`)](#-registos-logtxt)
- [Desbloqueio do Sistema](#-desbloqueio-do-sistema)
- [Comandos do Makefile](#-comandos-do-makefile)

---

## ✨ Funcionalidades Principais

1. **Bloqueio Inquebrável de Ecrã (X11 Low-Level Grab):**
   - Ecrã inteiro preto.
   - Captura direta de teclado e rato via `libX11` para impedir desvios (Alt+Tab, Super, Ctrl+Alt+T, etc.).
   - Desbloqueio silencioso ao digitar a sequência secreta (palavra-passe).

2. **Detecção de Movimento em Tempo Real & Filtro de Luz (Soft Light Filter):**
   - Analisa o fluxo da câmara webcam em background com OpenCV (`cv2.accumulateWeighted` e `cv2.absdiff`).
   - **Soft Light Filter**: Analisa a dispersão de pixéis para ignorar variações de iluminação global (sol, piscar de lâmpada) sem ignorar movimento humano.

3. **Reconhecimento Facial (YuNet + SFace ONNX):**
   - Utiliza modelos profundos de visão computacional da biblioteca OpenCV DNN:
     - **YuNet:** Detecção rápida e precisa de rostos.
     - **SFace:** Extração de *embeddings* (vetores de 128 dimensões) para comparação de características faciais.
   - Suporta formatos de imagem `.png`, `.jpg` e `.jpeg` na pasta `faces/`.
   - Suporta modo de saudação ativável/desativável (`greeting_enabled`).

4. **Registo Auditável de Eventos (`log.txt`):**
   - Regista todos os eventos com carimbo de data/hora no ficheiro `log.txt`.
   - Nenhuma imagem ou vídeo é gravado ou guardado em disco.

---

## 📂 Estrutura do Projeto

```text
/FaceRecSS/
├── FaceRecSS.py          # Script principal da aplicação (Tkinter + OpenCV + X11)
├── download_models.py    # Script para descarregar os modelos ONNX (YuNet + SFace)
├── Makefile              # Automação de tarefas (install, models, run, clean)
├── README.md             # Documentação do projeto
├── VersionControl.md     # Registo histórico de versões
├── log.txt               # Registo de eventos em texto
├── faces/                # Fotos de referência para pessoas reconhecidas (.png, .jpg, .jpeg)
└── models/               # Modelos ONNX (face_detection_yunet_2023mar.onnx e face_recognition_sface_2021dec.onnx)
```

---

## ⚙️ Configuração (`config.json`)

Opcionalmente, pode criar ou editar o ficheiro `config.json` na pasta do projeto para personalizar parâmetros:

```json
{
  "unlock_sequence": "unlock",
  "motion_threshold": 800,
  "camera_index": 0,
  "greeting_enabled": false,
  "recognition_threshold": 0.363,
  "recognition_cooldown": 10.0,
  "force_logout_minutes": 66,
  "font_family": "Helvetica",
  "default_greeting": "Olá {name}!\nNão devias estar a trabalhar?",
  "greetings": {
    "Nuno": "Olá Nuno! De volta ao trabalho!"
  }
}
```

---

## 📝 Registos (`log.txt`)

- **Ficheiro `log.txt`:** Regista carimbos de data/hora de eventos em formato texto:
  ```text
  2026-08-09 13:30:00 - FaceRec inicializado
  2026-08-09 13:30:00 - Sessão Iniciada / Ecrã Bloqueado
  2026-08-09 13:30:05 - Atividade detetada (Teclado)
  2026-08-09 13:30:10 - Face reconhecida: Nuno (confiança: 0.85) [Saudação: OFF]
  2026-08-09 13:35:10 - Sessão Desbloqueada com palavra-passe
  ```

---

## 🧰 Comandos do Makefile

| Comando | Descrição |
|---|---|
| `make install` | Instala dependências de Python (`opencv-python`, `pillow`, `numpy`) |
| `make models` | Executa o script `download_models.py` para descarregar os modelos ONNX |
| `make run` | Executa o `FaceRecSS.py` e limpa o ecrã no final |
| `make clean` | Elimina ficheiros temporários de compilação Python (`__pycache__`) |
| `make fclean` | Elimina ficheiros temporários e o ficheiro de registo `log.txt` |
| `make help` | Exibe a lista de comandos disponíveis |
