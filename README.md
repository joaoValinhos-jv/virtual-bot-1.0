VirtualBOT (VBOT) v1.0.0
AI Agent that performs desktop tasks or a queue of tasks by spoken or written commands.
In development  by João Vitor Souza Valinhos
Repository: https://github.com/joaoValinhos-jv/virtual-bot-1.0/blob/master/main.py

It uses the Gemini API, which means a Gemini API Key is needed, it is obtainable in:
https://aistudio.google.com/api-keys

Overview:
	Developed for saving time that would be spent by repeating steps that could be automated, for example: The Agent could open the browser, search for something, open the email, open desktop apps and even write a message about a specified topic while you fill your water bottle, answer a phone call or stay busy in another task.

Features:
	The system works by first sending the written or spoken sentence to Gemini -making use of the Gemini API- which has some previous instructions shown in the main.py file. After, the Gemini AI returns instructions or a reasonable answer for what was requested and the code figures how to use it, the returned information could be (x,y) mouse coordinates for clicking, a sentence for writing or searching in the web or some keyboard hotkeys that can do the ordered task.
	The built in AI knows what is happening on the screen according to the screenshot taken before the request, this screenshot goes to a buffer, so it is not allocated in the hard disk. Finally, to acknowledge that all the process happened successfully, a review function is called and a new screenshot is taken, if the process succeeds, then the next order must be requested, otherwise, the Agent tries again according to the number defined in the MAX_RETRIES macro. 
	An important information to talk about is that the speed of the requested steps is related to the version of the Gemini that is being used and to the amount of requests that the Gemini server is receiving, at some occasions there will emerge an error for high demand in the Gemini servers, in this cases simple tasks or queues may work but elaborated orders will not, so you must try to use the VBOT some time later.

Programming language: Python 3.14.7
Standard  Libraries: asyncio, io, os, time, re, json, sys
External Libraries: keyboard, numpy, pyautogui, mss, sounddevice, soundfile, python-dotenv, google-genai
API: Google Gemini API
UI/UX: CLI

Data processing:
	All the data is processed in the Gemini API, as well as the screenshots taken.

Authentication:
	A Gemini API Key must be written either in an “.env” file in the same directory that the executable file stands or it must be declared as an environmental variable.  

The “VirtualBOT.exe” file can be found in https://encurtador.com.br/FSio

