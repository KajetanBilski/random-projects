from mario import create_enviroment, DQNLightning

env = create_enviroment(render_mode="human")
model = DQNLightning.load_from_checkpoint("logs/lightning_logs/version_0/checkpoints/epoch=999-step=13000.ckpt")
net = model.net
agent = model.agent
agent.env = env

done = True
for step in range(5000):
    if done:
        env.reset()
    _, done = agent.play_step(net)

env.close()