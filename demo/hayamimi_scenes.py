"""Manim scenes for the hayamimi demo video.

Render with (from H:\\Programming\\Whisper-faster\\demo, using the dedicated venv):
  .venv-manim/Scripts/python -m manim -qh --fps 30 --media_dir manim_media hayamimi_scenes.py Intro Arch Outro
"""
from manim import *
import numpy as np
import random

# ---------------------------------------------------------------------------
# Shared visual identity
# ---------------------------------------------------------------------------
BG = "#0d0f13"
VERMILION = "#e04f2f"
CREAM = "#ece5d8"
DIM = "#b0a898"

JA = "#e04f2f"
EN = "#5b7fd4"
ZH = "#d4a13c"
KO = "#3fae9d"

MONO = "Consolas"
JP_BOLD = "Yu Mincho"

config.background_color = BG


class MonoText(Text):
    """Text mobject defaulting to the Consolas mono font.

    Used as DecimalNumber's mob_class since the default (MathTex) requires
    a LaTeX install we don't have on this machine.
    """

    def __init__(self, string, **kwargs):
        kwargs.setdefault("font", MONO)
        super().__init__(string, **kwargs)


# ---------------------------------------------------------------------------
# Scene 1 - intro.mp4 (~5s)
# ---------------------------------------------------------------------------
class Intro(Scene):
    def construct(self):
        self.camera.background_color = BG

        # Pulsing live indicator dot
        dot = Dot(radius=0.09, color=VERMILION).move_to(UP * 1.8)
        self.add(dot)

        def pulse(mob, alpha):
            s = 1.0 + 0.6 * np.sin(alpha * PI)
            mob.set(width=0.18 * s)
            mob.set_opacity(0.5 + 0.5 * (1 - alpha))

        self.play(UpdateFromAlphaFunc(dot, pulse, run_time=0.5))
        self.play(UpdateFromAlphaFunc(dot, pulse, run_time=0.5))
        self.play(FadeOut(dot, run_time=0.2))

        # Logotype 早耳
        haya = Text("早", font=JP_BOLD, weight=BOLD, color=CREAM)
        mimi = Text("耳", font=JP_BOLD, weight=BOLD, color=VERMILION)
        logo = VGroup(haya, mimi).arrange(RIGHT, buff=0.05).scale(2.6)
        logo.move_to(ORIGIN + UP * 0.3)

        self.play(
            LaggedStart(
                FadeIn(haya, shift=UP * 0.2),
                FadeIn(mimi, shift=UP * 0.2),
                lag_ratio=0.3,
            ),
            run_time=0.9,
        )

        # Letterspaced romanization
        roman = Text(
            "h a y a m i m i", font=MONO, color=DIM
        ).scale(0.55)
        roman.next_to(logo, DOWN, buff=0.35)
        self.play(FadeIn(roman, shift=UP * 0.1), run_time=0.5)

        # Thin horizontal rule draws itself
        rule = Line(LEFT * 1.6, RIGHT * 1.6, color=VERMILION, stroke_width=1.5)
        rule.next_to(roman, DOWN, buff=0.35)
        self.play(Create(rule), run_time=0.6)

        # Tagline
        tagline = Text(
            "その場で、聞き取る。", font=JP_BOLD, color=CREAM
        ).scale(0.5)
        tagline.next_to(rule, DOWN, buff=0.35)
        self.play(FadeIn(tagline), run_time=0.5)

        self.wait(1.0)


# ---------------------------------------------------------------------------
# Scene 2 - arch.mp4 (~11s)
# ---------------------------------------------------------------------------
class Arch(Scene):
    def construct(self):
        self.camera.background_color = BG
        random.seed(7)

        # ---- Waveform (left) ----
        n_bars = 40
        bars = VGroup()
        heights = []
        for i in range(n_bars):
            h = 0.15 + 0.9 * abs(np.sin(i * 0.7)) * (0.6 + 0.4 * random.random())
            heights.append(h)
            bar = RoundedRectangle(
                corner_radius=0.02,
                width=0.05,
                height=h,
                stroke_width=0,
                fill_color=CREAM,
                fill_opacity=0.85,
            )
            bars.add(bar)
        bars.arrange(RIGHT, buff=0.045, aligned_edge=DOWN)
        bars.scale(0.9)
        bars.move_to(LEFT * 5.2 + DOWN * 0.2)

        # fixed reference x-position (bottom-anchor) for each bar, captured
        # once so the wiggle updater below never compounds/drifts
        bar_bottom_y = bars.get_bottom()[1]
        bar_x_positions = [b.get_center()[0] for b in bars]
        scale_factor = 0.9

        self.play(
            LaggedStart(
                *[GrowFromEdge(b, DOWN) for b in bars],
                lag_ratio=0.02,
            ),
            run_time=1.2,
        )

        # gentle wiggle animation of the waveform while everything else builds
        def wiggle(mob, alpha):
            for i, b in enumerate(mob):
                h = heights[i] * scale_factor * (0.75 + 0.25 * np.sin(alpha * TAU * 2 + i * 0.5))
                h = max(h, 0.05)
                b.stretch_to_fit_height(h)
                b.move_to(
                    np.array([bar_x_positions[i], bar_bottom_y + h / 2, 0])
                )

        # VAD box
        vad = RoundedRectangle(
            corner_radius=0.12, width=1.6, height=1.0,
            stroke_color=CREAM, stroke_width=2, fill_opacity=0,
        )
        vad_label = Text("VAD", font=MONO, color=CREAM).scale(0.5)
        vad_group = VGroup(vad, vad_label).move_to(LEFT * 2.6)

        arrow1 = Arrow(
            bars.get_right(), vad_group.get_left(), buff=0.15,
            color=DIM, stroke_width=3, max_tip_length_to_length_ratio=0.15,
        )

        self.play(
            GrowArrow(arrow1),
            Create(vad),
            FadeIn(vad_label),
            run_time=0.9,
        )

        # LID diamond
        lid_shape = RegularPolygon(n=4, color=CREAM, stroke_width=2, fill_opacity=0)
        lid_shape.rotate(PI / 4).stretch(1.5, 0).stretch(1.0, 1)
        lid_label = Text("言語判定 LID", font=JP_BOLD, color=CREAM).scale(0.35)
        lid_group = VGroup(lid_shape, lid_label).move_to(LEFT * 0.2)

        arrow2 = Arrow(
            vad_group.get_right(), lid_group.get_left(), buff=0.15,
            color=DIM, stroke_width=3, max_tip_length_to_length_ratio=0.15,
        )

        self.play(
            GrowArrow(arrow2),
            Create(lid_shape),
            FadeIn(lid_label),
            run_time=0.9,
        )

        # ---- Four language boxes fanning out ----
        def make_lang_box(label_text, color, pos, font=JP_BOLD):
            box = RoundedRectangle(
                corner_radius=0.1, width=2.9, height=0.75,
                stroke_color=color, stroke_width=2.5, fill_color=color, fill_opacity=0.08,
            )
            label = Text(label_text, font=font, color=color).scale(0.32)
            label.move_to(box.get_center())
            grp = VGroup(box, label).move_to(pos)
            return grp

        ja_box = make_lang_box("日本語 ReazonSpeech", JA, RIGHT * 3.3 + UP * 2.4)
        en_box = make_lang_box("English/EU Parakeet", EN, RIGHT * 3.3 + UP * 0.8)
        zh_box = make_lang_box("中文 Paraformer", ZH, RIGHT * 3.3 + DOWN * 0.8)
        # Yu Mincho has no Hangul glyphs on this machine; Malgun Gothic covers
        # Korean (and still renders the Latin "SenseVoice" half cleanly).
        ko_box = make_lang_box(
            "한국어 SenseVoice", KO, RIGHT * 3.3 + DOWN * 2.4, font="Malgun Gothic"
        )

        lang_boxes = VGroup(ja_box, en_box, zh_box, ko_box)

        omni_box = RoundedRectangle(
            corner_radius=0.08, width=3.4, height=0.55,
            stroke_color=DIM, stroke_width=1.5, fill_opacity=0,
        )
        omni_label = Text(
            "+1600 languages Omnilingual", font=MONO, color=DIM
        ).scale(0.28)
        omni_label.move_to(omni_box.get_center())
        omni_group = VGroup(omni_box, omni_label).move_to(RIGHT * 3.3 + DOWN * 3.5)

        routes = VGroup(*[
            Arrow(
                lid_group.get_right(), box.get_left(), buff=0.15,
                color=box[0].get_stroke_color(), stroke_width=2.5,
                max_tip_length_to_length_ratio=0.08,
            )
            for box in lang_boxes
        ])
        omni_route = Line(
            lid_group.get_right(), omni_group.get_left(),
            color=DIM, stroke_width=1.5,
        )

        self.play(
            LaggedStart(
                *[GrowArrow(r) for r in routes],
                Create(omni_route),
                lag_ratio=0.15,
            ),
            run_time=1.3,
        )
        self.play(
            LaggedStart(
                *[FadeIn(box, shift=RIGHT * 0.2) for box in lang_boxes],
                FadeIn(omni_group),
                lag_ratio=0.15,
            ),
            run_time=1.1,
        )

        # ---- Pulse traveling the ja route + glow + label ----
        pulse_dot = Dot(radius=0.08, color=JA)
        pulse_dot.move_to(lid_group.get_right())

        self.add(pulse_dot)
        self.play(
            MoveAlongPath(pulse_dot, routes[0]),
            run_time=0.6,
            rate_func=rush_into,
        )

        glow = ja_box[0].copy().set_stroke(color=JA, width=6, opacity=1)
        confirm_text = Text(
            "〜0.1秒で確定", font=JP_BOLD, color=VERMILION
        ).scale(0.4)
        confirm_text.next_to(ja_box, UP, buff=0.2)

        self.play(
            FadeOut(pulse_dot, run_time=0.2),
            Flash(ja_box.get_center(), color=JA, line_length=0.3, num_lines=10, run_time=0.4),
            ja_box[0].animate.set_stroke(color=JA, width=4),
            FadeIn(confirm_text, shift=UP * 0.15),
            run_time=0.6,
        )

        # keep the waveform gently alive during the remaining hold
        self.play(UpdateFromAlphaFunc(bars, wiggle, run_time=2.0))
        self.play(UpdateFromAlphaFunc(bars, wiggle, run_time=1.7))

        self.wait(0.7)


# ---------------------------------------------------------------------------
# Scene 3 - outro.mp4 (~7s)
# ---------------------------------------------------------------------------
class Outro(Scene):
    def construct(self):
        self.camera.background_color = BG

        # Each stat is built as ONE Text mobject that is fully regenerated
        # every frame from the current tracker value (via always_redraw) and
        # re-centered on a fixed anchor point. This sidesteps the classic
        # DecimalNumber-plus-unit-label overlap bug, where a group is
        # arranged once at the starting value and the unit/label then drifts
        # out of alignment as the number grows extra digits.
        def counting_stat(tracker, anchor, fmt, t2c, sub_text):
            value_mob = always_redraw(
                lambda: Text(
                    fmt(tracker.get_value()),
                    font=MONO,
                    color=CREAM,
                    t2c=t2c,
                )
                .scale(1.5)
                .move_to(anchor)
            )
            sub_mob = Text(sub_text, font=JP_BOLD, color=DIM).scale(0.32)
            sub_mob.next_to(anchor, DOWN, buff=0.55)
            return value_mob, sub_mob

        # --- Stat 1: CER 5.8% ---
        cer_tracker = ValueTracker(0)
        cer_anchor = UP * 2.3
        cer_value, cer_sub = counting_stat(
            cer_tracker,
            cer_anchor,
            lambda v: f"CER {v:.1f}%",
            {"CER": DIM},
            "実放送日本語 (whisper-turbo 13.8%)",
        )

        # --- Stat 2: 100 ms ---
        ms_tracker = ValueTracker(0)
        ms_anchor = ORIGIN
        ms_value, ms_sub = counting_stat(
            ms_tracker,
            ms_anchor,
            lambda v: f"{v:.0f} ms",
            {},
            "発話終了から確定まで",
        )

        # --- Stat 3: 1600+ ---
        lang_tracker = ValueTracker(0)
        lang_anchor = DOWN * 2.3
        lang_value, lang_sub = counting_stat(
            lang_tracker,
            lang_anchor,
            lambda v: f"{v:.0f}+",
            {"+": VERMILION},
            "対応言語 / CPUのみ・GPU不要",
        )

        self.play(FadeIn(cer_value, shift=UP * 0.1), FadeIn(cer_sub), run_time=0.4)
        self.play(cer_tracker.animate.set_value(5.8), run_time=0.8, rate_func=rate_functions.ease_out_cubic)

        self.play(FadeIn(ms_value, shift=UP * 0.1), FadeIn(ms_sub), run_time=0.4)
        self.play(ms_tracker.animate.set_value(100), run_time=0.8, rate_func=rate_functions.ease_out_cubic)

        self.play(FadeIn(lang_value, shift=UP * 0.1), FadeIn(lang_sub), run_time=0.4)
        self.play(lang_tracker.animate.set_value(1600), run_time=0.8, rate_func=rate_functions.ease_out_cubic)

        self.wait(0.8)

        stats_group = VGroup(cer_value, cer_sub, ms_value, ms_sub, lang_value, lang_sub)
        self.play(FadeOut(stats_group), run_time=0.4)

        # --- Final card ---
        logo_small = Text("早耳", font=JP_BOLD, weight=BOLD, color=VERMILION).scale(1.1)
        repo = Text(
            "github.com/oboroge0/hayamimi", font=MONO, color=CREAM
        ).scale(0.5)
        mit = Text("MIT LICENSE", font=MONO, color=DIM).scale(0.35)

        final_card = VGroup(logo_small, repo, mit).arrange(DOWN, buff=0.3)
        self.play(FadeIn(final_card, shift=UP * 0.15), run_time=0.5)

        self.wait(1.5)
