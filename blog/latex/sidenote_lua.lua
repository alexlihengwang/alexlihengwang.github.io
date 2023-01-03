local counter = 0

function make_label(nonu)
  local label_class = 'margin-toggle' .. (nonu and '' or ' sidenote-number')
  local label_sym = nonu and '&#8853;' or ''
  local label_html = string.format(
    '<label for="sn-%d" class="%s">%s</label>',
    counter,
    label_class,
    label_sym
  )

  return pandoc.RawInline('html', label_html)
end

function make_checkbox(nonu)
  local input_html = string.format(
    '<input type="checkbox" id="sn-%d" class="margin-toggle"/>',
    counter
  )
  return pandoc.RawInline("html", input_html)
end

--- Convert footnotes into sidenotes
function Note (note)
  local inline_content = pandoc.utils.blocks_to_inlines(
    note.content,
    {pandoc.LineBreak()}
  )

  local nonu = false
  if inline_content[1] and inline_content[1].text == '{-}' then
    nonu = true
    table.remove(inline_content, 1)
  end

  local label = make_label(nonu)
  local input = make_checkbox(nonu)

  local note_type_class = nonu and "marginnote" or "sidenote"
  local note = pandoc.Span(
    inline_content,
    pandoc.Attr('', {note_type_class}, {})
  )

  counter = counter + 1

  return {label, input, note}
end