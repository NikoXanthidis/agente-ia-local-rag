from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Callable, Any

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QEvent
from PySide6.QtGui import QDesktopServices, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .agent import answer_question
from .config import DOCUMENTS_DIR, LLM_MODEL, EMBED_MODEL
from .conversation import (
    create_conversation,
    delete_conversation,
    get_messages,
    init_db,
    list_conversations,
)
from .ollama_client import check_ollama
from .rag import index_pdf, stats

#Esse arquivo é referente a interface gráfica do agente.

APP_STYLE = """
QMainWindow, QWidget {
    background: #111318;
    color: #e8eaf0;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 14px;
}
QFrame#Sidebar {
    background: #171a21;
    border-right: 1px solid #2a2e38;
}
QFrame#Header {
    background: #151820;
    border-bottom: 1px solid #2a2e38;
}
QPushButton {
    background: #252a35;
    border: 1px solid #343a47;
    border-radius: 8px;
    padding: 9px 12px;
    color: #f1f3f8;
}
QPushButton:hover { background: #303744; }
QPushButton:pressed { background: #1d222c; }
QPushButton#Primary {
    background: #2f6fed;
    border-color: #2f6fed;
    font-weight: 600;
}
QPushButton#Primary:hover { background: #3c79f0; }
QListWidget {
    background: transparent;
    border: 0;
    outline: 0;
}
QListWidget::item {
    border-radius: 7px;
    padding: 9px;
    margin: 2px 0;
}
QListWidget::item:selected { background: #2a3040; }
QListWidget::item:hover { background: #222733; }
QTextEdit#Input {
    background: #1b1f28;
    border: 1px solid #343a47;
    border-radius: 12px;
    padding: 10px;
    color: #f5f6fa;
}
QTextBrowser#UserMessage {
    background: #25375f;
    border: 1px solid #34508b;
    border-radius: 12px;
    padding: 9px;
}
QTextBrowser#AssistantMessage {
    background: #1b1f28;
    border: 1px solid #303642;
    border-radius: 12px;
    padding: 9px;
}
QLabel#Muted { color: #9aa3b2; }
QLabel#Brand { font-size: 18px; font-weight: 700; }
QLabel#Title { font-size: 16px; font-weight: 600; }
QStatusBar { background: #151820; color: #aeb6c4; }
QScrollArea { border: 0; background: #111318; }
QScrollBar:vertical { background: #171a21; width: 10px; }
QScrollBar::handle:vertical { background: #3a4050; border-radius: 5px; min-height: 30px; }
"""


class FunctionWorker(QThread):
    result = Signal(object)
    error = Signal(str)

    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.result.emit(self.fn(*self.args, **self.kwargs))
        except Exception as exc:
            self.error.emit(str(exc))


class MessageWidget(QWidget):
    def __init__(self, role: str, text: str, sources: list[dict] | None = None, parent=None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 6, 12, 6)

        card = QTextBrowser()
        card.setOpenExternalLinks(True)
        card.setReadOnly(True)
        card.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        card.setObjectName("UserMessage" if role == "user" else "AssistantMessage")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        card.setMaximumWidth(850)

        prefix = "**Você**\n\n" if role == "user" else "**Agente**\n\n"
        body = prefix + text

        if sources:
            source_lines = []
            for src in sources:
                arquivo = src.get("arquivo", "documento")
                pagina = src.get("pagina")
                source_lines.append(f"- {arquivo}" + (f" — página {pagina}" if pagina else ""))
            body += "\n\n---\n**Fontes recuperadas**\n" + "\n".join(source_lines)

        card.setMarkdown(body)
        card.document().setDocumentMargin(8)
        card.document().contentsChanged.connect(lambda: self._fit_height(card))
        QTimer.singleShot(0, lambda: self._fit_height(card))

        if role == "user":
            outer.addStretch(1)
            outer.addWidget(card)
        else:
            outer.addWidget(card)
            outer.addStretch(1)

    @staticmethod
    def _fit_height(card: QTextBrowser):
        width = max(card.viewport().width(), 300)
        card.document().setTextWidth(width)
        height = int(card.document().size().height()) + 24
        card.setFixedHeight(max(62, min(height, 900)))


class DesktopAgent(QMainWindow):
    def __init__(self):
        super().__init__()
        init_db()
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

        self.current_conversation_id: str | None = None
        self.workers: list[QThread] = []
        self.busy = False

        self.setWindowTitle("Agente IA Local — Regalia")
        self.resize(1280, 820)
        self.setMinimumSize(980, 650)
        self.setStyleSheet(APP_STYLE)

        self._build_ui()
        self._load_conversations(select_first=True)
        self._refresh_status()

        QShortcut(QKeySequence("Ctrl+N"), self, activated=self.new_conversation)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.import_pdf)

    # ----------------------------- UI -----------------------------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setMinimumWidth(245)
        sidebar.setMaximumWidth(330)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 16, 14, 14)
        side.setSpacing(9)

        brand = QLabel("Agente IA Local")
        brand.setObjectName("Brand")
        side.addWidget(brand)

        models = QLabel(f"{LLM_MODEL}  •  {EMBED_MODEL}")
        models.setObjectName("Muted")
        models.setWordWrap(True)
        side.addWidget(models)

        new_btn = QPushButton("＋ Nova conversa")
        new_btn.setObjectName("Primary")
        new_btn.clicked.connect(self.new_conversation)
        side.addWidget(new_btn)

        side.addWidget(QLabel("Conversas"))
        self.conversation_list = QListWidget()
        self.conversation_list.currentItemChanged.connect(self._conversation_selected)
        side.addWidget(self.conversation_list, 1)

        delete_btn = QPushButton("Excluir conversa")
        delete_btn.clicked.connect(self.delete_current_conversation)
        side.addWidget(delete_btn)

        side.addSpacing(8)
        side.addWidget(QLabel("Conhecimento"))

        pdf_btn = QPushButton("📄 Importar PDF")
        pdf_btn.clicked.connect(self.import_pdf)
        side.addWidget(pdf_btn)

        reindex_btn = QPushButton("↻ Reindexar PDFs")
        reindex_btn.clicked.connect(self.reindex_all_documents)
        side.addWidget(reindex_btn)

        folder_btn = QPushButton("Abrir pasta de documentos")
        folder_btn.clicked.connect(self.open_documents_folder)
        side.addWidget(folder_btn)

        self.knowledge_label = QLabel("Base: verificando...")
        self.knowledge_label.setObjectName("Muted")
        self.knowledge_label.setWordWrap(True)
        side.addWidget(self.knowledge_label)

        splitter.addWidget(sidebar)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("Header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 12, 18, 12)
        self.chat_title = QLabel("Nova conversa")
        self.chat_title.setObjectName("Title")
        header_layout.addWidget(self.chat_title)
        header_layout.addStretch(1)
        self.ollama_status = QLabel("● verificando Ollama")
        self.ollama_status.setObjectName("Muted")
        header_layout.addWidget(self.ollama_status)
        main_layout.addWidget(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.messages_layout = QVBoxLayout(self.scroll_content)
        self.messages_layout.setContentsMargins(18, 16, 18, 16)
        self.messages_layout.setSpacing(2)
        self.messages_layout.addStretch(1)
        self.scroll.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll, 1)

        input_wrap = QWidget()
        input_layout = QHBoxLayout(input_wrap)
        input_layout.setContentsMargins(18, 10, 18, 16)
        input_layout.setSpacing(10)

        self.input = QTextEdit()
        self.input.setObjectName("Input")
        self.input.setPlaceholderText("Digite sua mensagem...  (Enter envia, Shift+Enter quebra linha)")
        self.input.setMinimumHeight(58)
        self.input.setMaximumHeight(150)
        self.input.installEventFilter(self)
        input_layout.addWidget(self.input, 1)

        self.send_btn = QPushButton("Enviar")
        self.send_btn.setObjectName("Primary")
        self.send_btn.setMinimumWidth(100)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        main_layout.addWidget(input_wrap)
        splitter.addWidget(main)
        splitter.setSizes([275, 1005])

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Pronto")

    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if not (event.modifiers() & Qt.ShiftModifier):
                    self.send_message()
                    return True
        return super().eventFilter(obj, event)

    # ------------------------ conversations -----------------------
    def _load_conversations(self, select_first: bool = False, select_id: str | None = None):
        conversations = list_conversations()
        self.conversation_list.blockSignals(True)
        self.conversation_list.clear()

        for conv in conversations:
            item = QListWidgetItem(conv["title"])
            item.setData(Qt.UserRole, conv["id"])
            self.conversation_list.addItem(item)

        self.conversation_list.blockSignals(False)

        target_row = -1
        if select_id:
            for i in range(self.conversation_list.count()):
                item = self.conversation_list.item(i)
                if item.data(Qt.UserRole) == select_id:
                    target_row = i
                    break
        elif select_first and self.conversation_list.count():
            target_row = 0

        if target_row >= 0:
            self.conversation_list.setCurrentRow(target_row)
        elif not conversations:
            self.new_conversation()

    def _conversation_selected(self, current: QListWidgetItem | None, previous=None):
        if not current:
            return
        self.current_conversation_id = current.data(Qt.UserRole)
        self.chat_title.setText(current.text())
        self._render_history()

    def _render_history(self):
        self._clear_messages()
        if not self.current_conversation_id:
            return

        messages = get_messages(self.current_conversation_id)
        if not messages:
            self._add_welcome()
        else:
            for message in messages:
                self._add_message_widget(message["role"], message["content"])
        self._scroll_to_bottom()

    def new_conversation(self):
        conv = create_conversation()
        self.current_conversation_id = conv["id"]
        self._load_conversations(select_id=conv["id"])
        self.input.setFocus()

    def delete_current_conversation(self):
        if not self.current_conversation_id:
            return
        answer = QMessageBox.question(
            self,
            "Excluir conversa",
            "Excluir esta conversa e todo o histórico dela?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        delete_conversation(self.current_conversation_id)
        self.current_conversation_id = None
        self._load_conversations(select_first=True)

    # ----------------------------- chat ----------------------------
    def send_message(self):
        if self.busy:
            return
        text = self.input.toPlainText().strip()
        if not text:
            return

        if not self.current_conversation_id:
            self.new_conversation()

        conversation_id = self.current_conversation_id
        self.input.clear()
        self._add_message_widget("user", text)
        self._set_busy(True, "Pensando e consultando a base...")

        worker = FunctionWorker(answer_question, text, conversation_id)
        worker.result.connect(lambda result: self._chat_done(result, conversation_id))
        worker.error.connect(self._chat_error)
        self._run_worker(worker)

    def _chat_done(self, result: dict, conversation_id: str):
        self._set_busy(False, "Pronto")
        if conversation_id == self.current_conversation_id:
            self._add_message_widget(
                "assistant",
                result.get("answer", ""),
                result.get("sources", []),
            )
            self._scroll_to_bottom()
        self._load_conversations(select_id=self.current_conversation_id)

    def _chat_error(self, message: str):
        self._set_busy(False, "Erro")
        self._add_message_widget("assistant", f"**Erro:** {message}")
        QMessageBox.critical(self, "Erro ao consultar o agente", message)

    # -------------------------- documents --------------------------
    def import_pdf(self):
        if self.busy:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Importar documentos PDF",
            str(Path.home()),
            "Arquivos PDF (*.pdf)",
        )
        if not paths:
            return

        self._set_busy(True, f"Indexando {len(paths)} PDF(s)...")
        worker = FunctionWorker(self._import_and_index_files, paths)
        worker.result.connect(self._index_done)
        worker.error.connect(self._index_error)
        self._run_worker(worker)

    @staticmethod
    def _import_and_index_files(paths: list[str]) -> list[dict]:
        results = []
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        for raw_path in paths:
            src = Path(raw_path)
            dest = DOCUMENTS_DIR / src.name
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            results.append(index_pdf(dest))
        return results

    def reindex_all_documents(self):
        if self.busy:
            return
        files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
        if not files:
            QMessageBox.information(self, "Documentos", "Não há PDFs na pasta documentos.")
            return
        self._set_busy(True, f"Reindexando {len(files)} PDF(s)...")
        worker = FunctionWorker(lambda: [index_pdf(path) for path in files])
        worker.result.connect(self._index_done)
        worker.error.connect(self._index_error)
        self._run_worker(worker)

    def _index_done(self, results: list[dict]):
        self._set_busy(False, "Indexação concluída")
        total = sum(item.get("trechos_indexados", 0) for item in results)
        empty = sum(item.get("paginas_sem_texto", 0) for item in results)
        QMessageBox.information(
            self,
            "Indexação concluída",
            f"{len(results)} documento(s) processado(s).\n"
            f"{total} trechos indexados.\n"
            f"{empty} página(s) sem texto extraível.",
        )
        self._refresh_status()

    def _index_error(self, message: str):
        self._set_busy(False, "Erro na indexação")
        QMessageBox.critical(self, "Erro ao indexar PDF", message)

    def open_documents_folder(self):
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(DOCUMENTS_DIR.resolve())))

    # ---------------------------- status ---------------------------
    def _refresh_status(self):
        worker = FunctionWorker(self._status_payload)
        worker.result.connect(self._status_done)
        worker.error.connect(self._status_error)
        self._run_worker(worker)

    @staticmethod
    def _status_payload() -> dict:
        info = check_ollama()
        installed = [m.get("name", "") for m in info.get("models", [])]
        return {"installed": installed, "stats": stats()}

    def _status_done(self, payload: dict):
        installed = payload.get("installed", [])
        base = payload.get("stats", {}).get("trechos", 0)
        self.ollama_status.setText("● Ollama online")
        self.ollama_status.setStyleSheet("color: #55c781;")
        self.knowledge_label.setText(
            f"Base: {base} trechos\nModelo: {LLM_MODEL}\nEmbedding: {EMBED_MODEL}"
        )
        if not any(name.startswith(LLM_MODEL) for name in installed):
            self.statusBar().showMessage(f"Atenção: modelo {LLM_MODEL} não aparece no Ollama.")

    def _status_error(self, message: str):
        self.ollama_status.setText("● Ollama offline")
        self.ollama_status.setStyleSheet("color: #e56a6a;")
        self.knowledge_label.setText(f"Base: {stats().get('trechos', 0)} trechos")
        self.statusBar().showMessage(message)

    # --------------------------- helpers ---------------------------
    def _add_welcome(self):
        self._add_message_widget(
            "assistant",
            "Olá. Sou seu agente local de **TI e redes**. Posso manter o contexto da conversa, "
            "consultar os PDFs indexados e usar conhecimento geral de TI quando apropriado.\n\n"
            "Exemplos:\n"
            "- `Estou com um PC sem internet. O IP é 169.254.20.8`\n"
            "- `O que significa o erro E09102?`\n"
            "- `Qual comando mostra as portas abertas no Windows?`",
        )

    def _add_message_widget(self, role: str, text: str, sources: list[dict] | None = None):
        insert_at = max(0, self.messages_layout.count() - 1)
        self.messages_layout.insertWidget(insert_at, MessageWidget(role, text, sources))
        self._scroll_to_bottom()

    def _clear_messages(self):
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _scroll_to_bottom(self):
        QTimer.singleShot(50, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        ))

    def _set_busy(self, busy: bool, message: str):
        self.busy = busy
        self.send_btn.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.conversation_list.setEnabled(not busy)
        self.statusBar().showMessage(message)
        if not busy:
            self.input.setFocus()

    def _run_worker(self, worker: QThread):
        self.workers.append(worker)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        worker.start()

    def _cleanup_worker(self, worker: QThread):
        if worker in self.workers:
            self.workers.remove(worker)
        worker.deleteLater()


def run_desktop() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Agente IA Local Lite")
    app.setOrganizationName("Local AI")
    window = DesktopAgent()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_desktop())
