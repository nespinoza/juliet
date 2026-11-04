import matplotlib.pyplot as plt
import numpy as np
import starry
from astropy import units as u

starry.config.lazy = True
starry.config.quiet = True

star = starry.Primary(starry.Map(udeg=2, amp=1.0), m=1.0, r=1.0)
star.map[1:] = [0.5, 0.25]
planet = starry.Secondary(
    starry.Map(1, amp=0.0025), porb=1.0, r=0.1, prot=1000.0, m=0.0, t0=0.0, ecc=0.05, omega = 0.45*np.pi
)
planet.map[1, 0] = 0.5
sys1 = starry.System(star, planet, light_delay=True)
sys2 = starry.System(star, planet, light_delay=False)

n = 300

t_transit1 = np.linspace(-0.1, 0.1, n)
t_transit2 = np.linspace(-0.1,0.1, n) + 1.0
t_eclipse = np.linspace(-0.1, 0.1, n) + 0.5


# Light-delayed transit:
light_curve1 = sys1.flux(t_transit1).eval()
light_curve2 = sys1.flux(t_transit2).eval()
light_curve_eclipse = sys1.flux(t_eclipse).eval()

# Add noise:
sigma = 10*1e-6
noise1 = np.random.normal(0., sigma, len(light_curve1))
noise2 = np.random.normal(0., sigma, len(light_curve2))
noise3 = np.random.normal(0., sigma, len(light_curve_eclipse))

# Plot!
fig = plt.figure(figsize=(8, 3))
plt.plot(t_transit1, light_curve1, color = 'black', label = 'Light-travel time on')#, label="1st transit")
plt.plot(t_transit2, light_curve2, color = 'black')#, label="2nd transit")
plt.plot(t_eclipse, light_curve_eclipse, color = 'black')#, label="eclipse")

# Save:
fout = open('lttd-transit1.dat','w')
for i in range(len(t_transit1)):

    fout.write('{0:.10f} {1:.10f}\n'.format(t_transit1[i], light_curve1[i]))

fout.close()

fout = open('lttd-eclipse1.dat','w')
for i in range(len(t_eclipse)):

    fout.write('{0:.10f} {1:.10f}\n'.format(t_eclipse[i], light_curve_eclipse[i]))

fout.close()

fout = open('lttd-transit2.dat','w')
for i in range(len(t_transit2)):

    fout.write('{0:.10f} {1:.10f}\n'.format(t_transit2[i], light_curve2[i]))

fout.close()

# Try *not* light delay:
light_curve1 = sys2.flux(t_transit1).eval()
light_curve2 = sys2.flux(t_transit2).eval()
light_curve_eclipse = sys2.flux(t_eclipse).eval()

# Save:
fout = open('NOlttd-transit1.dat','w')
for i in range(len(t_transit1)):

    fout.write('{0:.10f} {1:.10f}\n'.format(t_transit1[i], light_curve1[i]))

fout.close()

fout = open('NOlttd-eclipse1.dat','w')
for i in range(len(t_eclipse)):

    fout.write('{0:.10f} {1:.10f}\n'.format(t_eclipse[i], light_curve_eclipse[i]))

fout.close()

fout = open('NOlttd-transit2.dat','w')
for i in range(len(t_transit2)):

    fout.write('{0:.10f} {1:.10f}\n'.format(t_transit2[i], light_curve2[i]))

fout.close()

# Plot!
plt.plot(t_transit1, light_curve1, color = 'grey', label = 'Light-travel time off')#, label="1st transit")
plt.plot(t_transit2, light_curve2, color = 'grey')#, label="2nd transit")
plt.plot(t_eclipse, light_curve_eclipse, color = 'grey')#, label="eclipse")
plt.xlabel("time [days]")
plt.ylabel("relative flux")
plt.legend(fontsize=10, loc="lower right")
_ = plt.title(
    "Light delay causes transits to occur early and eclipses late", fontsize=14
)

plt.show()
