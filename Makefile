# TiSoft - FaceRecSS (Screensaver com Deteção de Movimento e Reconhecimento Facial)

PYTHON ?= python3

all: install models run

help:
	@echo "Comandos disponíveis:"
	@echo "  make install  - Instala dependências Python (opencv-python, Pillow, numpy)"
	@echo "  make models   - Descarrega modelos ONNX (YuNet + SFace)"
	@echo "  make run      - Executa a aplicação FaceRecSS"
	@echo "  make clean    - Remove ficheiros temporários (__pycache__)"
	@echo "  make fclean   - Remove ficheiros temporários e registos (log.txt)"
	@echo "  make re       - Limpa e recompila (fclean -> all)"
	@echo "  make love     - Make love, not war"

install:
	$(PYTHON) -m pip install --user opencv-python pillow numpy

models:
	$(PYTHON) download_models.py

run:
	$(PYTHON) FaceRecSS.py

clean:
	rm -rf __pycache__ *.pyc *.pyo
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

fclean: clean
	rm -f log.txt

re: fclean all

love:
	@for f in " :-*          *-: " "  :-*        *-:  " "   :-*      *-:   " "    :-*    *-:    " "     :-*  *-:     " "      :-**-:      " "        💋      "; do \
		printf "\r%s" "$$f"; sleep 0.4; \
	done; echo ""

.PHONY: all help install models run clean fclean re love
