# Bolt Mini (work in progress)

A ~20 cm, ~300 g robot dog body: printed plastic frame, 8 MG90S micro servos
(2 per leg), a pan servo for the head and one for the tail, a toy shell or plush
cover as skin.

**Status: draft, do not print yet.** `cad/bolt_mini.scad` renders, but a
collision check still finds overlaps between the frame and the servos that need
fixing, and nothing has been printed or tested. The current plan is to base the
body on the proven open-source Sesame robot (github.com/dorianborian/sesame-robot)
and keep this file for Bolt's head mount (camera, OLED eyes, speaker).

Render a part:

    openscad -D 'part="frame"' -o frame.stl cad/bolt_mini.scad

Parts: `fit_test` (print first to check servo and horn fit), `frame`, `thigh`,
`shin`, `head`, and `assembly` (preview only).
