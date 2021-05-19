import pytorchtest


model = Model()
optimizer = 
pytorchtest.register(optimizer=optimizer)
pytorchtest.add_test(
    layer=,
    changing=True,
    range=[],
)
pytorchtest.add_unchanging_test(layer=)

for i in range(epochs):
    for batch in dataloader:
        output = model(batch['x'])
        loss = 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
