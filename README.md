# Mining-website
Chatbot to respond to text queries pertaining to various Acts, Rules, and Regulations applicable to Mining industries

Displays Information: The webpage shows information about the Ministry, its purpose, and the people in charge (Ministers).


Images and Carousels: It uses images of the Ministry's logo, important people, and coal mining activities. There's also a section with a carousel that automatically flips through different images.


Chatbot: There's a chat window at the bottom where you can type in questions and see responses (currently simulated).


Overall, this webpage aims to provide information and potentially answer questions through a chatbot in the future.


HTML Structure:

The core structure is built with HTML tags, defining elements like the header, navigation bar, content sections, and footer.




Bootstrap Integration:

Bootstrap classes are extensively used throughout the code to style the webpage and add responsiveness.
The navbar utilizes Bootstrap's navbar classes for a collapsible navigation bar that adapts to different screen sizes.
Grid classes (row and col) are employed to structure the content sections in a grid layout, ensuring a visually balanced layout.
The carousel section leverages Bootstrap's carousel component to create a rotating image slideshow with navigation controls.


Chatbot Functionality:

A basic chatbot interface is implemented using a designated chatbot-container section.
The chat window (chat-messages) displays messages with separate styling for user input (user-message) and bot responses (bot-message).
A text input field (chat-input) allows users to type their messages.
A button (send-button) triggers sending the user's message when clicked.


JavaScript Interaction:

JavaScript code is included to handle dynamic elements and user interactions.
A click event listener is added to the "Ask Queries" button to smoothly scroll down to the chatbot section upon clicking.
The chatbot functionality uses JavaScript functions to manage user input, display messages, and simulate bot responses (which can be replaced with a real chatbot engine later).


External Libraries:

The code incorporates Bootstrap from a CDN (Content Delivery Network) for easy access to the styling framework.
jQuery might be included (based on the provided code snippet) to simplify certain DOM (Document Object Model) manipulations for the chatbot.
Overall, this project demonstrates a well-structured webpage built with HTML and styled using Bootstrap. It integrates a basic chatbot interface with JavaScript for user interaction.
