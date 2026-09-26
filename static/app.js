const input = document.querySelector('#photo');
const area = document.querySelector('#selection-area');
const image = document.querySelector('#image-preview');
const wrap = document.querySelector('#image-wrap');
const selection = document.querySelector('#selection');
const form = document.querySelector('#upload-form');
const fields = ['crop_left', 'crop_top', 'crop_right', 'crop_bottom'].map(id => document.getElementById(id));
let origin = null;
let selected = false;
let objectUrl = null;

function clearSelection() {
  selected = false;
  selection.hidden = true;
  fields.forEach(field => field.value = '');
}

function point(event) {
  const box = image.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)),
    y: Math.max(0, Math.min(1, (event.clientY - box.top) / box.height))
  };
}

function draw(a, b) {
  const left = Math.min(a.x, b.x), top = Math.min(a.y, b.y);
  const right = Math.max(a.x, b.x), bottom = Math.max(a.y, b.y);
  selection.style.left = `${left * 100}%`;
  selection.style.top = `${top * 100}%`;
  selection.style.width = `${(right - left) * 100}%`;
  selection.style.height = `${(bottom - top) * 100}%`;
  selection.hidden = false;
  [left, top, right, bottom].forEach((value, index) => fields[index].value = value.toFixed(6));
  selected = right > left && bottom > top;
}

input.addEventListener('change', () => {
  clearSelection();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  if (!input.files.length) { area.hidden = true; return; }
  objectUrl = URL.createObjectURL(input.files[0]);
  image.src = objectUrl;
  area.hidden = false;
});

wrap.addEventListener('pointerdown', event => {
  if (!image.complete || !image.naturalWidth) return;
  event.preventDefault();
  clearSelection();
  origin = point(event);
  wrap.setPointerCapture(event.pointerId);
});
wrap.addEventListener('pointermove', event => {
  if (origin) draw(origin, point(event));
});
wrap.addEventListener('pointerup', event => {
  if (origin) draw(origin, point(event));
  origin = null;
});
document.querySelector('#clear-selection').addEventListener('click', clearSelection);
form.addEventListener('submit', event => {
  if (!selected) {
    event.preventDefault();
    alert('Marque a região da água antes de classificar.');
  }
});
