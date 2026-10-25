// The course's solid objects on screen, drawn from the same list the physics is built from (live.js
// setProps): every pole, post, wall, tunnel panel and platform appears exactly where and as big as it is
// solid. Looks: color, opacity, bands (stripes on a pole), cap (a ball on top of a pole).
import { propQuat } from "./live.js";

export function makeProps(THREE, scene, bolt, mesh, lin) {
  const group = new THREE.Group(); scene.add(group);
  const mats = {}, mat = (hex, opacity = 1) => mats[hex + opacity] || (mats[hex + opacity] = new THREE.MeshStandardMaterial({
    color: lin(hex), roughness: 0.6, transparent: opacity < 1, opacity, depthWrite: opacity >= 1, side: opacity < 1 ? THREE.DoubleSide : THREE.FrontSide }));
  let shown = null;
  function build(props) {
    while (group.children.length) group.remove(group.children[0]);
    for (const p of props || []) {
      const look = p.look || {}, m = mat(look.color || "#cccccc", look.opacity ?? 1), obj = new THREE.Group();
      if (p.kind === "cyl") {
        const [r, h] = p.size;
        obj.add(mesh(new THREE.CylinderGeometry(r, r, 2 * h, 20), m, [0, 0, 0], [Math.PI / 2, 0, 0]));
        if (look.bands) for (const z of [-0.5, 0.1, 0.7]) obj.add(mesh(new THREE.CylinderGeometry(r * 1.08, r * 1.08, 0.08, 20), mat(look.bands), [0, 0, z * h], [Math.PI / 2, 0, 0]));
        if (look.cap) obj.add(mesh(new THREE.SphereGeometry(r * 1.7, 16, 12), mat(look.cap), [0, 0, h + r]));
      } else {
        const [a, b, c] = p.size, box = mesh(new THREE.BoxGeometry(2 * a, 2 * b, 2 * c), m);
        if ((look.opacity ?? 1) < 1) box.castShadow = false;
        obj.add(box);
      }
      const q = propQuat(p);
      obj.position.set(...p.pos); obj.quaternion.set(q[1], q[2], q[3], q[0]);
      group.add(obj);
    }
  }
  return function draw() { if (bolt.props !== shown) { shown = bolt.props; build(shown); } };
}
