//#include <sys/types.h>
//#include <sys/socket.h>
//#include <sys/un.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <WinSock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <cstdio>
#include <cstdlib>
//#include <unistd.h>
#include "mainFunction.h"
#include "omp.h"

#define NAME ".socket"

#define DEFAULT_PORT "27015"

#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

int main()
{
	int sock, msgsock, rval;
	//struct sockaddr server;
	char buf[1024];
	char dummy[4];

	WSADATA wsaData;
	int iResult;
	// Initialize Winsock
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0)
	{
		printf("WSAStartup failed: %d\n", iResult);
		return 1;
	}

	struct addrinfo *result = NULL, *ptr = NULL, hints;

	ZeroMemory(&hints, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;
	hints.ai_flags = AI_PASSIVE;

	iResult = getaddrinfo(NULL, DEFAULT_PORT, &hints, &result);
	if (iResult != 0) {
		printf("getaddrinfo failed with error: %d\n", iResult);
		WSACleanup();
		return 1;
	}

	sock = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
	if (sock == INVALID_SOCKET)
	{
		printf("Error at socket(): %ld\n", WSAGetLastError());
		freeaddrinfo(result);
		WSACleanup();
		return 1;
	}
	//server.sun_family = AF_UNIX;
	//strcpy(server.sun_path, NAME);
	//unlink(NAME);
	iResult = bind(sock, result->ai_addr, (int)result->ai_addrlen);
	if (iResult == SOCKET_ERROR)
	{
		printf("bind failed with error: %d\n", WSAGetLastError());
		freeaddrinfo(result);
		closesocket(sock);
		WSACleanup();
		return 1;
	}
	freeaddrinfo(result);
	//printf("Socket has name %s\n", server.sun_path);
	//listen(sock, 5);
	printf("listen...\n");
	if (listen(sock, 5) == SOCKET_ERROR)
	{
		printf("Listen failed with error: %ld\n", WSAGetLastError());
		closesocket(sock);
		WSACleanup();
		return 1;
	}
	for (;;) {
		msgsock = accept(sock, NULL, NULL);
		if (msgsock == INVALID_SOCKET)
			perror("accept");
		else do {
			//bzero(buf, sizeof(buf));
			memset(buf, 0, sizeof(buf));
			rval = recv(msgsock, buf, 1024, 0);
			if (rval == -1)
				perror("reading stream message");
			else if (rval == 0)
				printf("Ending connection\n");
			else{
				double init = omp_get_wtime();
				mainFunction();
				printf("%f LACZNY CZAS\n", omp_get_wtime() - init);
				if (send(msgsock, dummy, sizeof(int),0) < 0 )
					perror("error writing stream socket");
				}
		} while (rval > 0);
		closesocket(msgsock);
	}
	closesocket(sock);
	//unlink(NAME);
	return 0;
}


