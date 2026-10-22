(() => {
  const tabs = Array.from(document.querySelectorAll('.report-tab'));

  const activateTab = (target, focus = false) => {
    tabs.forEach(tab => {
      const active = tab === target;
      tab.classList.toggle('is-active', active);
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      const panel = document.getElementById(tab.getAttribute('aria-controls'));
      if (panel) {
        panel.hidden = !active;
        panel.classList.toggle('is-active', active);
      }
    });
    if (focus) target.focus();
  };

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activateTab(tab));
    tab.addEventListener('keydown', event => {
      let next = index;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else return;
      event.preventDefault();
      activateTab(tabs[next], true);
    });
  });
  if (tabs.length) activateTab(tabs[0]);

  document.querySelectorAll('[data-kernel-picker]').forEach(select => {
    const container = document.getElementById(select.dataset.kernelPicker);
    if (!container) return;
    const items = Array.from(container.querySelectorAll('[data-kernel-item]'));
    const update = () => {
      items.forEach(item => { item.hidden = item.dataset.kernelItem !== select.value; });
    };
    select.addEventListener('change', update);
    update();
  });
})();
