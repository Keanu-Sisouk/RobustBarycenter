# fonction B alternative qui n'utilise pas pytorch, seulement pour p=2
def B2(y, p, q, its=250, lr=2, log=False, stop_threshold=1e-20):
    """
    Computes the barycenter images for candidate points x (n, d) and
    measure supports y: List(n, d_k).
    Output: (n, d) array
    """
    x = torch.zeros_like(y[0])
    #x = torch.randn(n, d, device=device, dtype=torch.double)
    #x.requires_grad_(True)
    loss_list = [1e10]
    #opt = Adam([x], lr=lr)
    exit_status = 'unknown'
    try:
        for _ in range(its):
            #opt.zero_grad()
            loss = torch.sum(C(x, y, p, q))
            #loss.backward()
            #opt.step()
            z = 0.
            x = torch.zeros_like(y[0])
            for k in range(len(y)):
                dk = torch.linalg.norm(y[k] - x)**(p-q)
                x += 1/len(y) * y[k]  / dk
                z += 1/len(y) * 1/ dk
            x = x / z   

            loss_list.append(loss.item())
            if stop_threshold > loss_list[-2] - loss_list[-1] >= 0:
                exit_status = 'Local optimum'
                raise StoppingCriterionReached
        exit_status = 'Max iterations reached'
        raise StoppingCriterionReached
    except StoppingCriterionReached:
        if log:
            return x, {'loss_list': loss_list[1:], 'exit_status': exit_status}
        return x