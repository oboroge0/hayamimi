import React from "react";
import { Composition } from "remotion";
import { Main } from "./Main";
import { DiarMain, DIAR_TOTAL_S } from "./DiarMain";
import {
  FPS,
  INTRO_SECONDS,
  ARCH_SECONDS,
  OUTRO_SECONDS,
  SECTIONS,
} from "./timeline";

const sectionsSeconds = SECTIONS.reduce(
  (acc, s) => acc + (s.displayEnd - s.displayStart),
  0
);

const TOTAL_SECONDS =
  INTRO_SECONDS + ARCH_SECONDS + sectionsSeconds + OUTRO_SECONDS;

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="Main"
        component={Main}
        durationInFrames={Math.round(TOTAL_SECONDS * FPS)}
        fps={FPS}
        width={1920}
        height={1080}
      />
      <Composition
        id="DiarMain"
        component={DiarMain}
        durationInFrames={Math.round(DIAR_TOTAL_S * FPS)}
        fps={FPS}
        width={1920}
        height={1080}
      />
    </>
  );
};
