import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 4.8))


x = [468]
y = [32.66]
l = ax.plot(x, y, '.', label='FSRCNN', ms=10, color='tab:blue')

e = [0.5,   0.7,   0.75,  0.8,  0.805, 0.815, 0.82,  0.825, 0.835, 1.5,   2.7,   3.45,  3.95,  4.0]
x = [89,    119,   145,   174,  182,   187,   190,   195,   196,   224,   225,   249,   286,   290]
y = [32.38, 32.45, 32.53, 32.6, 32.64, 32.66, 32.68, 32.69, 32.7,  32.71, 32.72, 32.73, 32.74, 32.75]
l = ax.plot(x, y, '-', label='ARM-FSRCNN', linewidth=3, color='tab:blue')



x = [1177]
y = [33.18]
l = ax.plot(x, y, '.', label='CARN', ms=10, color='tab:orange')

e = [0.2,   0.5,   1.0,   1.08,  1.14,  1.16,  1.2,   1.24,  1.26,  1.3,   1.32, 1.34,  1.36,  1.38,  1.5,      1.94,  ]
x = [240,   280,   290,   301,   311,   322,   353,   386,   421,   489,   504,  532,   601,   612,   683,      784,   ]
y = [32.93, 32.95, 32.98, 33.01, 33.03, 33.05, 33.08, 33.09, 33.11, 33.18, 33.2, 33.22, 33.26, 33.27, 33.3,     33.31, ]
l = ax.plot(x, y, '-', label='ARM-CARN', linewidth=3, color='tab:orange')



x = [5324]
y = [33.50]
l = ax.plot(x, y, '.', label='SRResNet', linewidth=3, ms=10, color='tab:red')

e = [0.2,   0.3,   0.5,   1.0,   1.4,   1.5,   1.6,  1.8,   1.9,   2.0,   4.0,   6.0,   7.5,   ]
x = [1048,  1218,  1220,  1243,  1295,  1321,  1352, 1391,  1520,  1832,  2406,  2869,  3243,  ]
y = [33.34, 33.35, 33.36, 33.37, 33.39, 33.39, 33.4, 33.41, 33.43, 33.46, 33.49, 33.51, 33.52, ]
l = ax.plot(x, y, '-', label='ARM-SRResNet', linewidth=3, color='tab:red')



lgd = ax.legend()



ax.set_xlabel('FLOPs (M)')
ax.set_ylabel('PSNR (dB)')
ax.grid(alpha=0.5)

fig.savefig('img/compare.png', bbox_inches='tight', pad_inches=0.05)