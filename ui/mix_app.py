from __future__ import annotations
import copy
import logging
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import ttk, filedialog, simpledialog, messagebox

from domain.models import Mix, Section
from file_io.mix_store import MixStore, DEFAULT_MIX_FILENAME
from playback.mix_engine import MixPlaybackEngine
from playback.transition_render import render_transition
from ui.mix_renderer import MixRenderer

logger = logging.getLogger(__name__)

_SOURCE_COLORS = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
    "#6a4c93", "#52b788", "#ef476f", "#118ab2", "#ffd166",
    "#06d6a0", "#fb8500", "#8ecae6", "#a8dadc", "#c77dff",
]


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
        self.selected: int | None = None  # anchor / focused section
        self._selection: set[int] = set()  # full multi-selection (always includes selected)
        self.dirty = False

        self._undo_stack: list = []
        self._redo_stack: list = []
        self._history_limit = 100

        self._engine = MixPlaybackEngine(
            on_section_change=self._on_playback_section_change,
            on_stop=self._on_playback_stop,
        )
        self._playing_sec: int | None = None

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
        edit_menu.add_separator()
        edit_menu.add_command(label="Merge with Selected", command=self._cmd_merge_selected,
                              accelerator="Ctrl+M")
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)

        cframe = tk.Frame(self.root, bg="#1a1a2e")
        cframe.pack(fill="both", expand=True, padx=10, pady=(4, 4))
        self.canvas = tk.Canvas(cframe, bg="#16213e", highlightthickness=0)
        vbar = ttk.Scrollbar(cframe, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._refresh())
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Control-Button-1>", self._on_ctrl_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        # Transport bar
        tframe = tk.Frame(self.root, bg="#111122", pady=6)
        tframe.pack(fill="x", padx=10, pady=(0, 10))
        self.play_btn = tk.Button(
            tframe, text="▶  Play", width=10, command=self._cmd_play,
            bg="#1e3a2e", fg="#44ff88", activebackground="#2a5a3a",
            relief="flat", font=("Segoe UI", 9, "bold"),
        )
        self.play_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = tk.Button(
            tframe, text="■  Stop", width=10, command=self._cmd_stop,
            bg="#1a1a2e", fg="#aaaaaa", activebackground="#2a2a3e",
            relief="flat", font=("Segoe UI", 9),
        )
        self.stop_btn.pack(side="left", padx=(0, 12))
        self.transport_label = tk.Label(
            tframe, text="", fg="#aaaaaa", bg="#111122", font=("Segoe UI", 9),
        )
        self.transport_label.pack(side="left")

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
        self.root.bind("<Control-m>", lambda e: self._cmd_merge_selected())
        self.root.bind("<space>", lambda e: self._cmd_play_stop_toggle())
        self.root.bind("<Escape>", lambda e: self._cmd_stop())

    def _refresh(self):
        if self.renderer is None:
            self.renderer = MixRenderer(self.canvas, self.mix)
        self.renderer.mix = self.mix
        self.renderer.draw_all(self.selected, playing_sec=self._playing_sec,
                               selection=self._selection)
        title = self.mix_path.name if self.mix_path else DEFAULT_MIX_FILENAME
        self.root.title(f"Mix Editor — {title}{' *' if self.dirty else ''}")
        self.name_label.config(text=self.mix.name)
        self.count_label.config(text=f"{len(self.mix.sections)} section(s)")

    # ------------------------------------------------------------------ events

    def _on_click(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        sec_idx = self.renderer.section_at_xy(cx, cy)
        self.selected = sec_idx
        self._selection = {sec_idx} if sec_idx is not None else set()
        self._refresh()

    def _on_ctrl_click(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        sec_idx = self.renderer.section_at_xy(cx, cy)
        if sec_idx is None:
            return
        if sec_idx in self._selection:
            self._selection.discard(sec_idx)
            if self.selected == sec_idx:
                self.selected = max(self._selection) if self._selection else None
        else:
            self._selection.add(sec_idx)
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
        can_merge = len(self._selection) >= 2
        menu.add_command(label="Merge with Selected", command=self._cmd_merge_selected,
                         state="normal" if can_merge else "disabled")
        menu.add_command(label="Transition with Selected", command=self._cmd_create_transition,
                         state="normal" if self._can_create_transition() else "disabled")
        menu.add_separator()
        menu.add_command(label="Play from Here", command=self._cmd_play,
                         state="normal" if self.mix.sections else "disabled")
        menu.add_command(label="Stop", command=self._cmd_stop,
                         state="normal" if self._engine.playing else "disabled")
        try:
            menu.post(event.x_root, event.y_root)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ undo/redo

    def _snapshot(self):
        return (copy.deepcopy(self.mix.sections), self.selected, frozenset(self._selection))

    def _push_undo(self):
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self._history_limit:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _restore(self, snapshot):
        sections, selected, selection = snapshot
        self.mix.sections = copy.deepcopy(sections)
        self._reindex()
        self._assign_bar_colors()
        self.selected = selected
        self._selection = set(selection)
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

    def _assign_bar_colors(self) -> None:
        """Assign Bar.color based on audiosource: same source → same color."""
        source_color: dict[str, str] = {}
        for sec in self.mix.sections:
            for bar in sec.bars:
                src = bar.audiosource
                if src not in source_color:
                    source_color[src] = _SOURCE_COLORS[len(source_color) % len(_SOURCE_COLORS)]
        for sec in self.mix.sections:
            sec.bars = [
                replace(bar, color=source_color.get(bar.audiosource, ""))
                for bar in sec.bars
            ]

    def _reindex(self):
        for i, sec in enumerate(self.mix.sections):
            sec.idx = i
        if self.selected is not None and self.mix.sections:
            self.selected = max(0, min(self.selected, len(self.mix.sections) - 1))
        elif not self.mix.sections:
            self.selected = None
        # Keep multi-selection in bounds
        self._selection = {i for i in self._selection if i < len(self.mix.sections)}
        if self.selected is not None:
            self._selection.add(self.selected)

    def _clear_selection(self):
        """Clear the multi-selection but preserve the anchor."""
        self._selection = {self.selected} if self.selected is not None else set()

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
        self._assign_bar_colors()
        self.selected = insert_at + len(new_secs) - 1
        self._clear_selection()
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
        if not self._selection:
            return
        self._push_undo()
        # Compute the anchor for the next selection before deletion
        next_idx = min(self._selection)
        # Delete in reverse order to preserve indices
        for idx in sorted(self._selection, reverse=True):
            del self.mix.sections[idx]
        self._reindex()
        if not self.mix.sections:
            self.selected = None
            self._selection = set()
        else:
            # Anchor to the first deleted position (adjusted for deletions before it)
            next_idx = min(next_idx, len(self.mix.sections) - 1)
            self.selected = next_idx
            self._selection = {next_idx}
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
        self._clear_selection()
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
        self._clear_selection()
        self.dirty = True
        self._refresh()

    def _can_create_transition(self) -> bool:
        if len(self._selection) != 2:
            return False
        idx_a, idx_b = sorted(self._selection)
        sec_a = self.mix.sections[idx_a]
        sec_b = self.mix.sections[idx_b]
        if not sec_a.bars or not sec_b.bars:
            return False
        if len(sec_a.bars) != len(sec_b.bars):
            return False
        src_a = {bar.audiosource for bar in sec_a.bars}
        src_b = {bar.audiosource for bar in sec_b.bars}
        return src_a.isdisjoint(src_b)

    def _cmd_create_transition(self):
        if not self._can_create_transition():
            return
        idx_a, idx_b = sorted(self._selection)
        sec_a = self.mix.sections[idx_a]
        sec_b = self.mix.sections[idx_b]
        transition_number = sum(1 for s in self.mix.sections if s.is_transition) + 1
        self._push_undo()
        transition = Section(
            idx=0,
            name=f"Transition - {transition_number}",
            bars=[],
            is_transition=True,
            fade_out_bars=copy.deepcopy(sec_a.bars),
            fade_in_bars=copy.deepcopy(sec_b.bars),
        )
        insert_at = idx_a + 1
        self.mix.sections.insert(insert_at, transition)
        self._reindex()
        self._render_transition_audio(transition)
        self._assign_bar_colors()
        self.selected = insert_at
        self._clear_selection()
        self.dirty = True
        self._refresh()

    def _render_transition_audio(self, transition: Section, out_dir: Path | None = None) -> None:
        """Render the overlaid mixed audio for *transition* and populate its bars.

        Updates ``transition.bars`` in place with the calculated beats and the
        generated mixed-audio source so they persist via MixStore.
        """
        out_dir = out_dir or (self.mix_path.parent if self.mix_path else Path.cwd())
        try:
            transition.bars = render_transition(transition, out_dir)
            if not transition.bars:
                messagebox.showwarning(
                    "Transition",
                    "Transition created, but no mixed audio could be rendered "
                    "(check the source WAV files).",
                )
        except Exception as e:
            logger.exception("Failed to render transition audio")
            messagebox.showerror("Transition", f"Could not render transition audio:\n{e}")

    def _cmd_merge_selected(self):
        if len(self._selection) < 2:
            return
        indices = sorted(self._selection)
        sections = [self.mix.sections[i] for i in indices]
        default_name = " + ".join(s.name for s in sections)
        new_name = simpledialog.askstring(
            "Merge Sections",
            "Name for merged section:",
            initialvalue=default_name,
            parent=self.root,
        )
        if new_name is None:
            return
        self._push_undo()
        merged_bars = []
        for s in sections:
            merged_bars.extend(s.bars)
        merged = Section(
            idx=sections[0].idx,
            name=new_name.strip() or default_name,
            bars=merged_bars,
        )
        lowest = indices[0]
        for i in reversed(indices):
            del self.mix.sections[i]
        self.mix.sections.insert(lowest, merged)
        self._reindex()
        self._assign_bar_colors()
        self.selected = lowest
        self._clear_selection()
        self.dirty = True
        self._refresh()

    # ------------------------------------------------------------------ playback commands

    def _cmd_play(self):
        """Play from selected section (or all sections if none selected)."""
        if not self.mix.sections:
            return
        start = self.selected if self.selected is not None else 0
        self._engine.play(self.mix.sections, start_sec=start)
        self._update_transport_ui()

    def _cmd_stop(self):
        self._engine.stop()
        self._playing_sec = None
        self._update_transport_ui()
        self._refresh()

    def _cmd_play_stop_toggle(self):
        if self._engine.playing:
            self._cmd_stop()
        else:
            self._cmd_play()

    def _on_playback_section_change(self, sec_idx: int):
        """Called from the worker thread when playback advances to a new section."""
        self._playing_sec = sec_idx
        self.root.after(0, self._update_transport_ui)
        self.root.after(0, self._refresh)

    def _on_playback_stop(self):
        """Called from the worker thread when playback ends naturally."""
        self._playing_sec = None
        self.root.after(0, self._update_transport_ui)
        self.root.after(0, self._refresh)

    def _update_transport_ui(self):
        if self._engine.playing:
            sec_idx = self._engine.current_section_idx
            if 0 <= sec_idx < len(self.mix.sections):
                name = self.mix.sections[sec_idx].name
                self.transport_label.config(text=f"Playing: {name}")
            self.play_btn.config(fg="#44ff88")
            self.stop_btn.config(fg="#ff6666")
        else:
            self.transport_label.config(text="")
            self.play_btn.config(fg="#44ff88")
            self.stop_btn.config(fg="#aaaaaa")

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
        self._selection = set()
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
        self._selection = set()
        self.dirty = False
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._ensure_transition_audio()
        self._assign_bar_colors()
        self._refresh()

    def _ensure_transition_audio(self) -> None:
        """Re-render transitions whose mixed audio is missing or stale."""
        for sec in self.mix.sections:
            if not sec.is_transition or not sec.fade_in_bars:
                continue
            wav = sec.bars[0].audiosource if sec.bars and sec.bars[0].beats else ""
            if sec.bars and wav and Path(wav).exists():
                continue
            self._render_transition_audio(sec)

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
            # Regenerate transition bars[] (calculated beats + generated audio
            # source) into the destination folder so the saved JSON is current.
            for sec in self.mix.sections:
                if sec.is_transition and sec.fade_in_bars and sec.fade_out_bars:
                    self._render_transition_audio(sec, out_dir=Path(path).parent)
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
        self._engine.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
