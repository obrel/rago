package backend

type Prompt struct {
	Messages   []Message  `json:"messages"`
	Parameters Parameters `json:"parameters"`
}

func NewPrompt() *Prompt {
	return &Prompt{}
}

func (p *Prompt) AddMessage(role, content string) *Prompt {
	p.Messages = append(p.Messages, Message{Role: role, Content: content})
	return p
}

func (p *Prompt) SetParameters(params Parameters) *Prompt {
	p.Parameters = params
	return p
}
