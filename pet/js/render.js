/* render.js: draws the simulated dog with three.js. Reads the simulation;
 * never changes it. */
(function (root) {
  'use strict';
  const THREE = root.THREE, P = root.PetPhysics;
  const { DOG, LEGS } = P;
  const UP = new THREE.Vector3(0, 1, 0);
  const v3 = a => new THREE.Vector3(a[0], a[1], a[2]);

  function cssColor(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return new THREE.Color(v || fallback);
  }

  class DogView {
    constructor(canvas) {
      this.canvas = canvas;
      this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
      this.renderer.setPixelRatio(Math.min(2, root.devicePixelRatio || 1));
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      this.renderer.outputEncoding = THREE.sRGBEncoding;
      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 60);
      this.orbit = { az: 0.5, el: 0.3, dist: 2.6, target: new THREE.Vector3(0, 0.3, 0) };
      this.blinkAt = 2;

      this.buildWorld();
      this.buildDog();
      this.applyTheme();
      this.bindOrbit();
      new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
      this.resize();
    }

    buildWorld() {
      const s = this.scene;
      this.hemi = new THREE.HemisphereLight(0xffffff, 0x8a8577, 0.75);
      s.add(this.hemi);
      const sun = this.sun = new THREE.DirectionalLight(0xffffff, 0.85);
      sun.position.set(1.5, 4, 2.5);
      sun.castShadow = true;
      sun.shadow.mapSize.set(1024, 1024);
      Object.assign(sun.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2, near: 0.5, far: 10 });
      s.add(sun, sun.target);

      this.floorMat = new THREE.MeshStandardMaterial({ roughness: 0.95 });
      const floor = new THREE.Mesh(new THREE.CircleGeometry(3.2, 72), this.floorMat);
      floor.rotation.x = -Math.PI / 2;
      floor.receiveShadow = true;
      s.add(floor);
      this.grid = new THREE.GridHelper(6.4, 32);
      this.grid.material.transparent = true;
      this.grid.material.opacity = 0.22;
      this.grid.position.y = 0.001;
      s.add(this.grid);

      // The path of the body centre through the air.
      this.traceGeom = new THREE.BufferGeometry();
      this.traceGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(600 * 3), 3));
      this.trace = new THREE.Line(this.traceGeom, new THREE.LineDashedMaterial({ color: 0xff7a1a, dashSize: 0.03, gapSize: 0.02 }));
      this.trace.frustumCulled = false;
      s.add(this.trace);
    }

    buildDog() {
      const metal = this.metal = new THREE.MeshStandardMaterial({ color: 0xdfe2e8, metalness: 0.3, roughness: 0.42 });
      const dark = new THREE.MeshStandardMaterial({ color: 0x353a45, metalness: 0.2, roughness: 0.6 });
      const accent = new THREE.MeshStandardMaterial({ color: 0xff7a1a, metalness: 0.1, roughness: 0.5 });
      this.eyeMat = new THREE.MeshBasicMaterial({ color: 0x7cd4ff });
      this.mouthMat = new THREE.MeshBasicMaterial({ color: 0x222630 });
      const box = (w, h, d, mat, x = 0, y = 0, z = 0) => {
        const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
        m.position.set(x, y, z); m.castShadow = true; return m;
      };
      const ball = (r, mat) => { const m = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 12), mat); m.castShadow = true; return m; };

      const dog = this.dog = new THREE.Group();
      dog.add(box(DOG.bodyLength, DOG.bodyHeight, DOG.bodyWidth, metal));
      dog.add(box(0.36, 0.012, 0.15, dark, 0, DOG.bodyHeight / 2 + 0.006, 0));
      this.batteryBar = box(0.3, 0.014, 0.03, new THREE.MeshBasicMaterial({ color: 0x40c057 }), 0, DOG.bodyHeight / 2 + 0.014, 0);
      dog.add(this.batteryBar);
      LEGS.forEach(l => { const j = ball(0.036, accent); j.position.set(...l.hip); dog.add(j); });

      // Head on a neck pivot so it can nod and tilt.
      const head = this.headPivot = new THREE.Group();
      head.position.set(DOG.bodyLength / 2 - 0.02, 0.07, 0);
      head.add(box(0.17, 0.13, 0.17, metal, 0.07, 0.06, 0));
      head.add(box(0.08, 0.07, 0.11, metal, 0.18, 0.025, 0));
      head.add(box(0.02, 0.026, 0.045, dark, 0.226, 0.05, 0));
      this.mouth = box(0.012, 0.012, 0.06, this.mouthMat, 0.222, 0.0, 0);
      head.add(this.mouth);
      this.eyes = [-1, 1].map(s => {
        const g = new THREE.Group();
        g.position.set(0.157, 0.085, 0.045 * s);
        const open = box(0.01, 0.036, 0.03, this.eyeMat);
        open.castShadow = false;
        const cross = new THREE.Group();
        [-1, 1].forEach(r => { const b = box(0.01, 0.045, 0.008, this.eyeMat); b.rotation.x = r * Math.PI / 4; cross.add(b); });
        g.add(open, cross);
        g.userData = { open, cross };
        head.add(g);
        return g;
      });
      this.ears = [-1, 1].map(s => {
        const e = box(0.03, 0.08, 0.045, dark, 0.02, 0.16, 0.06 * s);
        e.rotation.x = 0.25 * s;
        head.add(e);
        return e;
      });
      dog.add(head);

      // Tail on a pivot so it can wag.
      const tailGeom = new THREE.CylinderGeometry(0.01, 0.016, 0.2, 10);
      tailGeom.translate(0, 0.1, 0);
      const tail = new THREE.Mesh(tailGeom, dark);
      tail.castShadow = true;
      this.tailPivot = new THREE.Group();
      this.tailPivot.position.set(-DOG.bodyLength / 2, 0.05, 0);
      this.tailPivot.add(tail);
      dog.add(this.tailPivot);
      this.scene.add(dog);

      // Legs live in world space; each frame we stretch them between hip, knee and foot.
      const cyl = (r, mat) => { const g = new THREE.CylinderGeometry(r, r, 1, 14); const m = new THREE.Mesh(g, mat); m.castShadow = true; this.scene.add(m); return m; };
      this.legMeshes = LEGS.map(() => ({
        upper: cyl(0.028, metal), lower: cyl(0.022, metal),
        knee: (() => { const k = ball(0.032, accent); this.scene.add(k); return k; })(),
        foot: (() => { const f = ball(0.03, dark); this.scene.add(f); return f; })(),
      }));
    }

    applyTheme() {
      const bg = cssColor('--scene-bg', '#e9e6df');
      this.scene.background = bg;
      this.scene.fog = new THREE.Fog(bg, 4, 9);
      this.floorMat.color = cssColor('--scene-floor', '#f4f2ec');
      this.grid.material.color = cssColor('--scene-grid', '#9a958a');
    }

    bindOrbit() {
      const c = this.canvas, pts = new Map();
      let pinch = 0;
      c.addEventListener('pointerdown', e => { c.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); });
      const end = e => { pts.delete(e.pointerId); pinch = 0; };
      c.addEventListener('pointerup', end);
      c.addEventListener('pointercancel', end);
      c.addEventListener('pointermove', e => {
        if (!pts.has(e.pointerId)) return;
        const [px, py] = pts.get(e.pointerId);
        pts.set(e.pointerId, [e.clientX, e.clientY]);
        if (pts.size === 1) {
          this.orbit.az -= (e.clientX - px) * 0.008;
          this.orbit.el = P.clamp(this.orbit.el + (e.clientY - py) * 0.006, 0.05, 1.35);
        } else if (pts.size === 2) {
          const [a, b] = [...pts.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
          if (pinch) this.zoom(pinch / d);
          pinch = d;
        }
      });
      c.addEventListener('wheel', e => { e.preventDefault(); this.zoom(Math.exp(e.deltaY * 0.001)); }, { passive: false });
      c.addEventListener('dblclick', () => Object.assign(this.orbit, { az: 0.5, el: 0.3, dist: 2.6 }));
    }
    zoom(f) { this.orbit.dist = P.clamp(this.orbit.dist * f, 0.9, 5.5); }

    resize() {
      const p = this.canvas.parentElement, w = p.clientWidth, h = p.clientHeight;
      if (!w || !h) return;
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }

    // Where a world point lands on screen, in CSS pixels of the canvas.
    project(world) {
      const v = v3(world).project(this.camera);
      const r = this.canvas.getBoundingClientRect();
      return { x: (v.x + 1) / 2 * r.width, y: (1 - v.y) / 2 * r.height, visible: v.z < 1 };
    }

    cameraXZ() { return [this.camera.position.x, this.camera.position.z]; }

    render(sim, extras, realDt) {
      const t = sim.time, o = this.orbit;
      // Camera follows the dog smoothly, ignoring the bounce of a jump.
      const k = 1 - Math.exp(-realDt * 3);
      o.target.x += (sim.pos[0] - o.target.x) * k;
      o.target.z += (sim.pos[2] - o.target.z) * k;
      o.target.y += (0.3 + Math.max(0, sim.pos[1] - DOG.standHeight) * 0.6 - o.target.y) * k * 2;
      this.camera.position.set(
        o.target.x + o.dist * Math.cos(o.el) * Math.sin(o.az),
        o.target.y + o.dist * Math.sin(o.el),
        o.target.z + o.dist * Math.cos(o.el) * Math.cos(o.az));
      this.camera.lookAt(o.target);
      this.sun.position.set(o.target.x + 1.5, 4, o.target.z + 2.5);
      this.sun.target.position.copy(o.target);

      // Body
      this.dog.position.set(...sim.pos);
      this.dog.rotation.set(sim.roll, sim.yaw, sim.pitch, 'YZX');
      this.batteryBar.scale.x = Math.max(0.02, extras.battery / 100);
      this.batteryBar.position.x = -0.15 + 0.15 * this.batteryBar.scale.x;
      this.batteryBar.material.color.setHex(extras.battery > 30 ? 0x40c057 : extras.battery > 12 ? 0xfab005 : 0xfa5252);

      // Head, ears, eyes, mouth
      this.headPivot.rotation.set(sim.head.tilt, 0, sim.head.pitch);
      const ex = sim.expression, sad = ex === 'sad' || ex === 'sleep';
      this.ears.forEach((e, i) => { e.rotation.z = sad ? -0.9 : -0.1 + Math.sin(t * 3 + i) * 0.03 * sim.excitement; });
      if (t > this.blinkAt + 0.12) this.blinkAt = t + 2 + Math.random() * 3;
      const blinking = t > this.blinkAt && ex === 'normal';
      const eyeY = { happy: 0.35, sleep: 0.1, sad: 0.6, dizzy: 1 }[ex] ?? (blinking ? 0.1 : 1);
      this.eyes.forEach((g, i) => {
        g.userData.open.visible = ex !== 'dead';
        g.userData.cross.visible = ex === 'dead';
        g.userData.open.scale.y = eyeY;
        g.rotation.x = ex === 'dizzy' ? t * 12 * (i ? 1 : -1) : 0;
      });
      if (sim.speaking) {
        this.mouthMat.color.setHex(0xff7a1a);
        this.mouth.scale.y = 1 + 2.2 * Math.abs(Math.sin(t * 17));
      } else {
        this.mouthMat.color.setHex(ex === 'happy' ? 0xff7a1a : 0x222630);
        this.mouth.scale.y = 1;
      }

      // Tail: wags faster and higher when excited, droops when sad.
      const exc = Math.min(1.4, sim.excitement);
      const wag = Math.sin(t * (5 + 11 * exc)) * (0.1 + 0.5 * exc);
      this.tailPivot.rotation.set(0, wag, sad ? 2.3 : 0.75 - 0.25 * exc, 'YZX');

      // Legs
      sim.legs.forEach((leg, i) => {
        const m = this.legMeshes[i];
        this.stretch(m.upper, leg.hip, leg.knee);
        this.stretch(m.lower, leg.knee, leg.foot);
        m.knee.position.set(...leg.knee);
        m.foot.position.set(...leg.foot);
      });

      // Flight path
      const pts = sim.trace.slice(-600), arr = this.traceGeom.attributes.position.array;
      pts.forEach((p, i) => arr.set(p, i * 3));
      this.traceGeom.setDrawRange(0, pts.length);
      this.traceGeom.attributes.position.needsUpdate = true;
      this.trace.computeLineDistances();
      this.trace.visible = pts.length > 1 && extras.showTrace;

      this.renderer.render(this.scene, this.camera);
    }

    stretch(mesh, a, b) {
      const A = v3(a), B = v3(b), d = B.clone().sub(A), L = d.length();
      mesh.position.copy(A).addScaledVector(d, 0.5);
      mesh.scale.set(1, Math.max(L, 1e-3), 1);
      if (L > 1e-6) mesh.quaternion.setFromUnitVectors(UP, d.normalize());
    }
  }

  root.PetView = { DogView };
})(self);
