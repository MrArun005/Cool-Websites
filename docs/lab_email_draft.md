# Draft: asking a robotics lab to test our brain on a real quadruped

Fill in the [brackets] before sending. Send it from your own email, to one person at a time (a professor or lab
manager whose group works on legged robots; IITs, IISc and IIITs often have a Unitree Go2/A1).

---

**Subject:** Request: 1-hour test of a dog-motion controller on your Unitree [Go2 / A1]

Dear Prof. [Name],

I'm [your name], an independent developer from [city]. Over the past few weeks I have built, with AI assistance,
a simulated robot dog that learned real dog movement from motion-capture data. All of it runs in MuJoCo with
A1-class motor limits (33.5 N·m, 420 W, 21 rad/s, with back-EMF braking modelled).

Results so far, all in simulation, on fair tests with fresh random seeds:
- It copies 8 dog motions (walk, trot, pace, turns, sit, lie down, get up front-first) at 91–97%, and the canter at
  76% with no falls in 90 tries.
- It clears a 50 cm agility bar from a run-up in 77% of tries.
- One network ("MasterMind", trained by DAgger from the expert controllers) switches between all of these by
  itself, succeeding at 86.6% of switches.

I am now [retraining these controllers on the official Unitree [Go2/A1] MuJoCo model and testing them under sensor
noise and control delay]. I would be very grateful for the chance to run them on your robot for about an hour,
under your lab's safety rules: walking and trotting first on a tether or harness, with your team holding the
stop.

I can share the code, the trained policies and the full test reports beforehand, and credit your lab in anything
I publish. The dog motion data is CC BY-NC 4.0, so this is non-commercial work.

Would you or someone in your group be open to this? I can come to your lab at a time that suits you.

Thank you for your time,
[Your name]
[Phone] · [Email] · [GitHub link]

---

Notes before sending:
- Only send once the Go2/A1 retraining and the noise/delay tests are done, so the numbers in [brackets] are real.
- Keep the "in simulation" wording; never claim real-robot results before the test.
- Attach or link the demo page only if it is shared publicly (Share menu on the artifact).
