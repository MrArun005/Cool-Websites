# Bolt Duck: shopping list

From the official Open Duck Mini v2 BOM (prices in euros, as listed there; links in the
[original spreadsheet](https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit?usp=sharing)).
Prices vary by country and shop; check before buying.

## Walking robot (€398, about €331 at the lowest prices)

### Power
| Part | Qty | € each | € total | Note |
|---|---|---|---|---|
| 18650 Li-ion cell (e.g. Molicel P30B, 3000 mAh, 30 A) | 2 | 5 | 10 | Get high-discharge cells |
| 18650 2-cell holder | 1 | 4.99 | 4.99 | |
| 2S BMS (battery protection board) | 1 | 8.40 | 8.40 | |
| 5 V regulator (UBEC, 4–6 A) | 1 | 4 | 4 | Powers the Pi |
| Small power switch | 1 | 4.49 | 4.49 | |
| USB-C charger module (2S) | 1 | 9.99 | 9.99 | |
| 2.1 mm barrel jack | 2 | 1 | 2 | |
| XT30 connector pair | 1 | 8 | 8 | Sold in packs |

### Motors, control, sensors
| Part | Qty | € each | € total | Note |
|---|---|---|---|---|
| **Feetech STS3215 servo, 7.4 V (19 kg·cm)** | **14** | 14 | 196 | Make sure it's the 7.4 V version |
| Waveshare bus servo control board | 1 | 5 | 5 | Talks to all 14 servos over one wire |
| Raspberry Pi Zero 2 W | 1 | 26.08 | 26.08 | The robot's computer |
| microSD card | 1 | 10 | 10 | |
| BNO055 IMU (balance sensor) | 1 | 40 | 40 | Cheaper clones exist; untested |
| Foot contact switches (SS-10) | 4 | 1 | 4 | |
| 9 g micro servos (antennas) | 2 | 3.33 | 6.66 | |

### Hardware & materials
| Part | Qty | € total | Note |
|---|---|---|---|
| M3 heat-set inserts (pack) | 1 | 5.99 | |
| M3 screws (assorted, 6–10 mm) | 1 | – | Not in the official BOM; buy a small M3 kit |
| Bearings | 3 | 10.50 | Check the size in the assembly guide |
| PLA filament | ~1 kg | 20 | See PRINT.md |
| TPU filament | a little | – | Foot soles only |
| Cable sheath | 1 | 9 | |
| Micro-USB to USB-C cables (15 cm + 50 cm) | 2 | 13 | |
| Threadlocker (Loctite 243) | 1 | – | For metal-to-metal screws |

## Pet package: what makes it Bolt (+€34)

| Part | Qty | € total | Note |
|---|---|---|---|
| Speaker | 1 | 6 | Talks |
| Amplifier (I2S, e.g. MAX98357A) | 1 | 5 | Drives the speaker |
| Microphone | 1 | 15 | Hears you. The duck project hadn't wired this yet; Bolt needs it |
| Raspberry Pi camera | 1 | 5 | Sees you, for following. Also not wired yet in the original |
| LEDs (eyes + projector) | 3 | 0.60 | Buy a pack |
| Eye diffusers | 4 | 0.64 | |
| Projector reflector | 1 | 2 | The "flashlight" |

**Total with the pet package: about €432** (€365 at the lowest prices), plus screws, threadlocker and shipping.

## Tools
Screwdrivers (hex keys for M3), soldering iron (for heat-set inserts and battery wiring),
multimeter, wire strippers, small pliers. A 3D printer or a print service (see PRINT.md).

## Safety
Lithium cells: use the BMS, never short the cells, charge with the proper 2S charger, don't
leave it charging unattended. Servos are strong: keep fingers out of joints when testing.
