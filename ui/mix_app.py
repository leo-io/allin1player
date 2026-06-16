from __future__ import annotations
import copy
import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, simpledialog, messagebox

from domain.models import Mix
from file_io.mix_store import MixStore, DEFAULT_MIX_FILENAME
from ui.mix_renderer import MixRenderer

logger = logging.getLogger(__name__)


class MixEditor:
    """Section-block editor for a Mix (the "mix-editor").

    Arranges whole sections (imported from existing arrangements) as blocks and
    persists them to a `mix-editor.json` file. Mirrors ArrangementEditor's
    structure (canvas + menus + snapshot undo/redo) but has no audio playback.
    """

    def __init__(self, mix_path: str | None = None):
        self.mix: Mix = Mix(name="New Mix")
        self.mix_path: Path | None = None
        self.renderer: MixRenderer | None = None
        self.selected: int | None = None
        self.dirty = False

        self._undo_stack: list = []
        self._redo_stack: list = []
        self._history_limit = 100

        self._build_ui()

        if mix_path and Path(mix_path).exists():
            self.load_file(mix_path)
        else:
            self._refresh()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Mix Editor")
        self.root.configure(bg="#1a1a2e")
        self.root.minsize(900, 480)

        top = tk.Frame(self.root, bg="#1a1a2e")
        top.pack(fill="x", padx=16, pady=(14, 4))
        self.name_label = tk.Label(top, font=("Segoe UI", 11, "bold"),
                                   fg="#e0e0e0", bg="#1a1a2e")
        self.name_label.pack(side="left")
        self.count_label = tk.Label(top, font=("Segoe UI", 11),
                                    fg="#aaaaaa", bg="#1a1a2e")
        self.count_label.pack(side="right")

        self.menu_bar = tk.Menu(self.root, bg="#1a1a2e", fg="#e0e0e0")
        self.root.config(menu=self.menu_bar)

        file_menu = tk.Menu(self.menu_bar, tearoff=0, bg="#1a1a2e", fg="#e0e0e0")
        file_menu.add_command(label="New", command=self._cmd_new, accelerator="Ctrl+N")
        file_menu.add_command(label="Open…", command=self._cmd_open, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self._cmd_save, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As…", command=self._cmd_save_as)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(self.menu_bar, tearoff=0, bg="#1a1a2e", fg="#e0e0e0")
        edit_menu.add_command(label="Undo", command=self._cmd_undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self._cmd_redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Add Section…", command=self._cmd_add_section,
                              accelerator="Ins")
        edit_menu.add_command(label="Rename Section", command=self._cmd_rename,
                              accelerator="F2")
        edit_menu.add_command(label="Delete Section", command=self._cmd_delete,
                              accelerator="Del")
        edit_menu.add_separator()
        edit_menu.add_command(label="Move Left", command=self._cmd_move_left,
                              accelerator="Ctrl+←")
        edit_menu.add_command(label="Move Right", command=self._cmd_move_right,
                              accelerator="Ctrl+→")
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)

        cframe = tk.Frame(self.root, bg="#1a1a2e")
        cframe.pack(fill="both", expand=True, padx=10, pady=(4, 14))
        self.canvas = tk.Canvas(cframe, bg="#16213e", highlightthickness=0)
        vbar = ttk.Scrollbar(cframe, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._refresh())
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind("<Control-n>", lambda e: self._cmd_new())
        self.root.bind("<Control-o>", lambda e: self._cmd_open())
        self.root.bind("<Control-s>", lambda e: self._cmd_save())
        self.root.bind("<Control-z>", lambda e: self._cmd_undo())
        self.root.bind("<Control-y>", lambda e: self._cmd_redo())
        self.root.bind("<Control-Shift-Z>", lambda e: self._cmd_redo())
        self.root.bind("<Delete>", lambda e: self._cmd_delete())
        self.root.bind("<Insert>", lambda e: self._cmd_add_section())
        self.root.bind("<F2>", lambda e: self._cmd_rename())
        self.root.bind("<Control-Left>", lambda e: self._cmd_move_left())
        self.root.bind("<Control-Right>", lambda e: self._cmd_move_right())

    def _refresh(self):
        if self.renderer is None:
            self.renderer = MixRenderer(self.canvas, self.mix)
        self.renderer.mix = self.mix
        self.renderer.draw_all(self.selected)
        title = self.mix_path.name if self.mix_path else DEFAULT_MIX_FILENAME
        self.root.title(f"Mix Editor — {title}{' *' if self.dirty else ''}")
        self.name_label.config(text=self.mix.name)
        self.count_label.config(text=f"{len(self.mix.sections)} section(s)")

    # ------------------------------------------------------------------ events

    def _on_click(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        sec_idx = self.renderer.section_at_xy(cx, cy)
        self.selected = sec_idx
        self._refresh()

    def _on_right_click(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        sec_idx = self.renderer.section_at_xy(cx, cy)
        if sec_idx is not None:
            self.selected = sec_idx
            self._refresh()
        menu = tk.Menu(self.root, tearoff=0, bg="#1a1a2e", fg="#e0e0e0")
        menu.add_command(label="Add Section…", command=self._cmd_add_section)
        has_sel = self.selected is not None
        menu.add_command(label="Rename", command=self._cmd_rename,
                         state="normal" if has_sel else "disabled")
        menu.add_command(label="Delete", command=self._cmd_delete,
                         state="normal" if has_sel else "disabled")
        menu.add_separator()
        menu.add_command(label="Move Left", command=self._cmd_move_left,
                         state="normal" if has_sel else "disabled")
        menu.add_command(label="Move Right", command=self._cmd_move_right,
                         state="normal" if has_sel else "disabled")
        try:
            menu.post(event.x_root, event.y_root)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ undo/redo

    def _snapshot(self):
        return (copy.deepcopy(self.mix.sections), self.selected)

    def _push_undo(self):
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self._history_limit:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _restore(self, snapshot):
        sections, selected = snapshot
        self.mix.sections = copy.deepcopy(sections)
        self._reindex()
        self.selected = selected
        self.dirty = True
        self._refresh()

    def _cmd_undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())

    def _cmd_redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())

    def _reindex(self):
        for i, sec in enumerate(self.mix.sections):
            sec.idx = i
        if self.selected is not None and self.mix.sections:
            self.selected = max(0, min(self.selected, len(self.mix.sections) - 1))
        elif not self.mix.sections:
            self.selected = None

    # ------------------------------------------------------------------ editing commands

    def _cmd_add_section(self):
        path = filedialog.askopenfilename(
            title="Choose an arrangement to import a section from",
            filetypes=[("Arrangement JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            sections = MixStore.list_arrangement_sections(path)
        except Exception as e:
            messagebox.showerror("Import Error", f"Could not load arrangement:\n{e}")
            return
        if not sections:
            messagebox.showinfo("Import", "That arrangement has no sections.")
            return
        chosen = self._choose_sections(sections)
        if not chosen:
            return
        self._push_undo()
        insert_at = (self.selected + 1) if self.selected is not None else len(self.mix.sections)
        new_secs = [copy.deepcopy(sections[i]) for i in chosen]
        self.mix.sections[insert_at:insert_at] = new_secs
        self._reindex()
        self.selected = insert_at + len(new_secs) - 1
        self.dirty = True
        logger.info(f"Added {len(new_secs)} section(s) from {path}")
        self._refresh()

    def _choose_sections(self, sections) -> list[int]:
        """Modal picker: returns indices of selected sections (multi-select)."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Select section(s) to add")
        dlg.configure(bg="#1a1a2e")
        dlg.transient(self.root)
        dlg.grab_set()
        tk.Label(dlg, text="Select one or more sections:", bg="#1a1a2e",
                 fg="#e0e0e0", font=("Segoe UI", 10)).pack(padx=12, pady=(12, 6))
        lb = tk.Listbox(dlg, selectmode="extended", width=40, height=min(15, len(sections)),
                        bg="#16213e", fg="#e0e0e0", selectbackground="#2a3a5a")
        for sec in sections:
            lb.insert("end", f"{sec.name}  [{sec.bar_count()} bars]")
        lb.pack(padx=12, pady=6, fill="both", expand=True)
        lb.selection_set(0)

        result: list[int] = []

        def on_ok():
            result.extend(lb.curselection())
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        btns = tk.Frame(dlg, bg="#1a1a2e")
        btns.pack(pady=(6, 12))
        tk.Button(btns, text="Add", command=on_ok, width=10).pack(side="left", padx=6)
        tk.Button(btns, text="Cancel", command=on_cancel, width=10).pack(side="left", padx=6)
        lb.bind("<Double-Button-1>", lambda e: on_ok())

        dlg.wait_window()
        return list(result)

    def _cmd_rename(self):
        if self.selected is None:
            return
        sec = self.mix.sections[self.selected]
        new_name = simpledialog.askstring("Rename Section", f"New name for '{sec.name}':",
                                          initialvalue=sec.name, parent=self.root)
        if new_name and new_name.strip():
            self._push_undo()
            sec.name = new_name.strip()
            self.dirty = True
            self._refresh()

    def _cmd_delete(self):
        if self.selected is None:
            return
        self._push_undo()
        del self.mix.sections[self.selected]
        self._reindex()
        if not self.mix.sections:
            self.selected = None
        else:
            self.selected = min(self.selected, len(self.mix.sections) - 1)
        self.dirty = True
        self._refresh()

    def _cmd_move_left(self):
        if self.selected is None or self.selected == 0:
            return
        self._push_undo()
        i = self.selected
        self.mix.sections[i - 1], self.mix.sections[i] = self.mix.sections[i], self.mix.sections[i - 1]
        self.selected = i - 1
        self._reindex()
        self.dirty = True
        self._refresh()

    def _cmd_move_right(self):
        if self.selected is None or self.selected >= len(self.mix.sections) - 1:
            return
        self._push_undo()
        i = self.selected
        self.mix.sections[i + 1], self.mix.sections[i] = self.mix.sections[i], self.mix.sections[i + 1]
        self.selected = i + 1
        self._reindex()
        self.dirty = True
        self._refresh()

    # ------------------------------------------------------------------ file commands

    def _cmd_new(self):
        if not self._confirm_discard():
            return
        name = simpledialog.askstring("New Mix", "Mix name:",
                                      initialvalue="New Mix", parent=self.root)
        if name is None:
            return
        self.mix = Mix(name=name.strip() or "New Mix")
        self.mix_path = None
        self.selected = None
        self.dirty = False
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._refresh()

    def _cmd_open(self):
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(
            title="Open mix",
            filetypes=[("Mix JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.load_file(path)

    def load_file(self, path: str):
        try:
            self.mix = MixStore.load(path)
        except Exception as e:
            messagebox.showerror("Open Error", f"Could not load mix:\n{e}")
            return
        self.mix_path = Path(path)
        self.selected = None
        self.dirty = False
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._refresh()

    def _cmd_save(self):
        if self.mix_path is None:
            self._cmd_save_as()
            return
        self._do_save(self.mix_path)

    def _cmd_save_as(self):
        path = filedialog.asksaveasfilename(
            title="Save mix as",
            defaultextension=".json",
            initialfile=DEFAULT_MIX_FILENAME,
            filetypes=[("Mix JSON", "*.json")],
        )
        if path:
            self._do_save(Path(path))

    def _do_save(self, path: Path):
        try:
            MixStore.save(self.mix, path)
            self.mix_path = path
            self.dirty = False
            self._refresh()
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save mix:\n{e}")

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        result = messagebox.askyesnocancel("Unsaved Changes", "Save current mix first?")
        if result is None:
            return False
        if result:
            self._cmd_save()
        return True

    def quit(self):
        if not self._confirm_discard():
            return
        self.root.destroy()

    def run(self):
        self.root.mainloop()
