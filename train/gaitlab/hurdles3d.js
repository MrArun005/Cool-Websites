// Real agility hurdles on screen, drawn from the physics (hurdle.py / live.js placeReal): each frame the
// stands are put where their mocap bodies are and each bar where its free body is, so what you see is
// exactly what is solid, and a knocked bar falls on screen as it does in the physics.
// Parts below the floor (the stands reach 75 cm under the bar so any height fits) are hidden by the floor.

export function makeHurdles(THREE, scene, bolt, mesh, lin) {
  const g = bolt.c.real_hurdle;
  if (!g) return () => {};
  const std = (hex, extra = {}) => new THREE.MeshStandardMaterial({ color: lin(hex), roughness: 0.6, ...extra });
  const white = std("#f4f4f0"), blue = std("#2f6fb5"), red = std("#c4372f"), yellow = std("#e8b62c");
  const items = bolt.real.map(() => {
    const stand = new THREE.Group(), bar = new THREE.Group();
    const postH = g.post_top - g.bottom, wingH = g.wing_top - g.bottom, [w0, w1] = g.wing_y;
    for (const s of [1, -1]) {
      // the upright the cups hang on, and the wing panel outside it, framed like a real wing
      stand.add(mesh(new THREE.CylinderGeometry(g.post_r, g.post_r, postH, 20), white, [0, s * g.post_y, g.bottom + postH / 2], [Math.PI / 2, 0, 0]));
      stand.add(mesh(new THREE.BoxGeometry(2 * g.wing_t, w1 - w0, wingH), blue, [0, s * (w0 + w1) / 2, g.bottom + wingH / 2]));
      stand.add(mesh(new THREE.CylinderGeometry(0.018, 0.018, wingH + 0.06, 16), white, [0, s * w1, g.bottom + (wingH + 0.06) / 2], [Math.PI / 2, 0, 0]));
      for (const z of [g.wing_top - 0.03, g.wing_top - 0.22]) stand.add(mesh(new THREE.BoxGeometry(2 * g.wing_t + 0.004, w1 - w0, 0.025), white, [0, s * (w0 + w1) / 2, z]));
      stand.add(mesh(new THREE.BoxGeometry(2 * g.wing_t + 0.006, 0.05, 0.05), yellow, [0, s * (w0 + w1) / 2, g.wing_top - 0.12], [0.785, 0, 0]));
      // the cup and its two lips
      const [cx, cy, cz] = g.cup, [lx, ly, lz] = g.lip;
      stand.add(mesh(new THREE.BoxGeometry(2 * cx, 2 * cy, 2 * cz), white, [0, s * g.cup_y, g.cup_top - cz]));
      for (const f of [1, -1]) stand.add(mesh(new THREE.BoxGeometry(2 * lx, 2 * ly, 2 * lz), white, [f * g.lip_x, s * g.cup_y, g.cup_top + lz]));
    }
    // the bar: red and white bands, along its local y like the physics capsule
    const n = 6, len = (2 * g.bar_half) / n;
    for (let i = 0; i < n; i++) bar.add(mesh(new THREE.CylinderGeometry(g.bar_r, g.bar_r, len, 16), i % 2 ? white : red, [0, -g.bar_half + len * (i + 0.5), 0]));
    for (const s of [1, -1]) bar.add(mesh(new THREE.SphereGeometry(g.bar_r, 16, 12), s > 0 ? white : red, [0, s * g.bar_half, 0]));
    scene.add(stand, bar);
    return { stand, bar };
  });
  const d = bolt.data;
  return function draw() {
    bolt.real.forEach((H, k) => {
      const { stand, bar } = items[k];
      stand.visible = bar.visible = !!H.at;
      if (!H.at) return;
      const p = d.mocap_pos, q = d.mocap_quat, m = 3 * H.mocap, n = 4 * H.mocap;
      stand.position.set(p[m], p[m + 1], p[m + 2]); stand.quaternion.set(q[n + 1], q[n + 2], q[n + 3], q[n]);
      const [bp, bq] = bolt.barPose(k);
      bar.position.set(bp[0], bp[1], bp[2]); bar.quaternion.set(bq[1], bq[2], bq[3], bq[0]);
    });
  };
}
