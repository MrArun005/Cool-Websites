// Bolt Mini: printable parts for a ~20 cm robot dog.
//
// Render one part at a time:
//   openscad -D 'part="frame"' -o frame.stl bolt_mini.scad
// Parts: fit_test, frame, thigh, shin, head, assembly (preview only).
//
// Print in PLA or PETG, 0.2 mm layers, 30% infill, 4 walls. No supports needed:
// every part prints flat on the face it is drawn on.
//
// Units: millimetres. Frame coordinates: +x forward, +y left, +z up.
// Measure your servos with calipers and adjust the numbers below if they differ.

part = "assembly";

// ---------- servo: MG90S / SG90 class micro servo ----------
servo_len   = 22.8;   // body length (along the mounting tabs)
servo_wid   = 12.2;   // body width
servo_shaft = 5.9;    // shaft centre, measured from the near end of the body
tab_span    = 32.2;   // tab tip to tab tip
tab_holes   = 27.8;   // mounting hole spacing
tab_height  = 16.0;   // body height below the tabs (the part that goes through a pocket)
above_tabs  = 12.5;   // tab underside to top of the output shaft
fit         = 0.4;    // clearance added to pockets; raise if too tight

pilot   = 1.8;        // pilot hole for the servo's self-tapping screws
m3pilot = 2.6;        // pilot for M3 screws into plastic

// ---------- horn: single-arm servo horn ----------
horn_hub = 7.6;       // hub diameter (+ clearance)
horn_tip = 4.6;       // arm width at the tip (+ clearance)
horn_len = 15.0;      // hub centre to arm tip
horn_dep = 1.6;       // recess depth
horn_holes = [9, 13]; // pilot holes along the arm for horn screws

// ---------- dog geometry (must match the simulation model) ----------
body_len  = 150;   // frame length
body_wid  = 64;    // frame width, outside of the walls
wall      = 3;
deck      = 2;
wall_h    = 28;
hip_x     = 55;    // hips at x = ±55
hip_z     = 15;    // hip axis height in the frame
thigh_len = 50;    // hip axis to knee axis
shin_len  = 55;    // knee axis to centre of the toe
plate_t   = 3.5;   // leg plate thickness
head_x    = 93;    // head pan servo shaft, clear of the front crossbar
tail_z    = 22;    // tail servo shaft height

$fn = 48;

// A pocket a servo body drops through, with its shaft at the origin and the body
// running along +x. Includes the two tab screw holes.
module servo_pocket(depth = 20) {
    translate([-servo_shaft - fit / 2, -servo_wid / 2 - fit / 2, -depth / 2])
        cube([servo_len + fit, servo_wid + fit, depth]);
    for (s = [-1, 1])
        translate([servo_len / 2 - servo_shaft + s * tab_holes / 2, 0, -depth / 2])
            cylinder(d = pilot, h = depth);
}

// Recess for a single-arm horn, arm pointing along -y, cut into the top of a plate at z = top.
module horn_recess(top) {
    translate([0, 0, top - horn_dep])
        linear_extrude(horn_dep + 1)
            hull() { circle(d = horn_hub); translate([0, -horn_len]) circle(d = horn_tip); }
    translate([0, 0, -1]) cylinder(d = 6.5, h = top + 2);           // screwdriver access to the horn screw
    for (y = horn_holes) translate([0, -y, -1]) cylinder(d = 1.5, h = top + 2);
}

// Print this first: one pocket and one horn recess, to check your servo and horn fit.
module fit_test() {
    difference() {
        translate([-12, -32, 0]) cube([24, 50, plate_t]);
        translate([0, -20, 0]) rotate(-90) servo_pocket(20);
        translate([0, 12, 0]) horn_recess(plate_t);
    }
}

// Body frame: bottom deck, two side walls holding the four hip servos (shafts pointing
// outward), corner posts for the top plate, a front ledge for the head servo (shaft up)
// and a rear crossbar for the tail servo (shaft pointing back).
module frame() {
    difference() {
        union() {
            translate([-body_len / 2, -body_wid / 2, 0]) cube([body_len, body_wid, deck]);
            for (s = [-1, 1]) translate([-body_len / 2, s > 0 ? body_wid / 2 - wall : -body_wid / 2, 0])
                cube([body_len, wall, wall_h]);
            for (x = [-1, 1], y = [-1, 1]) translate([x * 70, y * 25, 0]) cylinder(d = 7, h = wall_h);
            translate([70, -body_wid / 2, 20]) cube([5, body_wid, 8]);             // front crossbar
            translate([70, -16, wall_h - 3]) cube([head_x + 12 - 70, 32, 3]);       // head ledge
            translate([-75, -body_wid / 2, deck]) cube([5, body_wid, wall_h - deck]); // rear wall
        }
        // hip servo pockets: shaft at (±hip_x, hip_z), body running toward the middle of the dog
        for (x = [-1, 1], y = [-1, 1])
            translate([x * hip_x, y * (body_wid / 2 - wall / 2), hip_z])
                rotate([90, 0, 0]) rotate([0, 0, x > 0 ? 180 : 0]) servo_pocket(wall + 2);
        // head pan servo: shaft up, body hanging under the ledge in front of the frame
        translate([head_x, 0, wall_h - 1.5]) rotate([0, 0, 180]) servo_pocket(5);
        // tail servo: shaft pointing back, body standing upright between the rear hips
        translate([-72.5, 0, tail_z]) rotate([0, 90, 0]) servo_pocket(7);
        // screw holes for the top plate
        for (x = [-1, 1], y = [-1, 1]) translate([x * 70, y * 25, deck]) cylinder(d = m3pilot, h = wall_h);
        // weight-saving and cable slots in the deck
        for (x = [-30, 0, 30]) translate([x - 10, -18, -1]) cube([20, 36, deck + 2]);
    }
}

// Thigh: hip horn recess at the top, knee servo pocket at the bottom.
module thigh() {
    difference() {
        linear_extrude(plate_t) hull() {
            circle(r = 9);
            translate([-10, -thigh_len - servo_shaft - 6]) square([20, servo_len + 6]);
        }
        horn_recess(plate_t);
        translate([0, -thigh_len, 0]) rotate(90) servo_pocket(plate_t * 3);
    }
}

// Shin: knee horn recess at the top, a round toe at the bottom. Slide a 10 mm silicone
// tube, a rubber chair-leg tip or a cut toy ball over the toe for grip.
module shin() {
    difference() {
        union() {
            linear_extrude(plate_t) hull() {
                circle(r = 8);
                translate([0, -shin_len]) circle(r = 5);
            }
            translate([0, -shin_len, 0]) cylinder(d = 10, h = 10);
        }
        horn_recess(plate_t);
    }
}

// Head: a base that sits on the pan servo horn, and a face plate for a 0.96" OLED
// (the eyes) above a Raspberry Pi camera. Zip-tie slots hold a toy head shell.
oled_window = [25, 13];
oled_holes  = 23.5;   // hole spacing of common 0.96" SSD1306 modules; check yours
cam_holes   = [21, 12.5];
module head() {
    difference() {
        union() {
            translate([-20, -17, 0]) cube([40, 34, 3]);
            translate([17, -20, 0]) cube([3, 40, 60]);
            for (y = [-15, 12]) translate([5, y, 3]) cube([12, 3, 12]);          // gussets
        }
        translate([0, 0, 3]) mirror([0, 0, 1]) horn_recess(3);   // recess on the underside, onto the horn
        translate([16, 0, 0]) {
            translate([0, -oled_window[0] / 2, 44 - oled_window[1] / 2]) cube([5, oled_window[0], oled_window[1]]);
            for (y = [-1, 1], z = [-1, 1]) translate([0, y * oled_holes / 2, 44 + z * oled_holes / 2]) rotate([0, 90, 0]) cylinder(d = 2.2, h = 5);
            translate([0, 0, 17]) rotate([0, 90, 0]) cylinder(d = 8, h = 5);
            for (y = [-1, 1], z = [-1, 1]) translate([0, y * cam_holes[0] / 2, 17 + z * cam_holes[1] / 2]) rotate([0, 90, 0]) cylinder(d = 2.2, h = 5);
        }
        for (x = [-15, 9], y = [-1, 1]) translate([x, y * 13 - 2, -1]) cube([4, 4, 5]);  // zip-tie slots
    }
}

// ---------- assembly preview (not for printing) ----------
module servo_dummy() {   // shaft at origin pointing +z, tabs at z = 0
    color("DimGray") translate([-servo_shaft, -servo_wid / 2, -tab_height]) cube([servo_len, servo_wid, tab_height + 10]);
    color("DimGray") translate([servo_len / 2 - servo_shaft - tab_span / 2, -servo_wid / 2, -2.5]) cube([tab_span, servo_wid, 2.5]);
    color("Gold") cylinder(d = 4.8, h = above_tabs);
}

hip_deg  = 30;   // standing pose, same idea as the simulation's
knee_deg = 60;
module leg(front, side) {
    // side: +1 left, -1 right. The leg hangs in the x-z plane outside the wall,
    // horn recesses facing the body. The right leg is the left one mirrored.
    k = front ? -1 : 1;   // front knees point back, rear knees forward
    mirror([0, side > 0 ? 0 : 1, 0])
        translate([0, body_wid / 2 + above_tabs + plate_t, 0]) rotate([90, 0, 0])
            rotate([0, 0, k * hip_deg]) {
                color("Orange") thigh();
                translate([0, -thigh_len, 0]) {
                    mirror([0, 0, 1]) rotate([0, 0, 90]) servo_dummy();   // knee servo, shaft pointing out
                    translate([0, 0, -above_tabs - plate_t]) rotate([0, 0, -2 * k * hip_deg]) color("Orange") shin();
                }
            }
}

module assembly() {
    color("WhiteSmoke") frame();
    for (x = [-1, 1], y = [-1, 1])
        translate([x * hip_x, y * (body_wid / 2), hip_z]) rotate([y * -90, 0, 0]) rotate([0, 0, x > 0 ? 180 : 0]) servo_dummy();
    for (x = [-1, 1], y = [-1, 1]) translate([x * hip_x, 0, hip_z]) leg(x > 0, y);
    color("SaddleBrown", 0.85) translate([-75, -32, wall_h + 6]) cube([145, 64, 3]);   // hand-cut top plate
    translate([head_x, 0, wall_h]) { rotate([0, 0, 180]) servo_dummy(); translate([0, 0, above_tabs]) color("Orange") head(); }
    translate([-72.5, 0, tail_z]) rotate([0, -90, 0]) rotate([0, 0, 180]) servo_dummy();          // tail servo
    color("DimGray") translate([-30, -17, deck]) cube([60, 34, 18]);                              // battery
}

if (part == "fit_test") fit_test();
else if (part == "frame") frame();
else if (part == "thigh") thigh();
else if (part == "shin") shin();
else if (part == "head") head();
else assembly();
